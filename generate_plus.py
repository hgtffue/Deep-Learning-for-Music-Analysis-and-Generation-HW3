"""Improved sampling for Stable Audio Open (optional part of HW3).

Adds, on top of generate.py:
  * --guidance apg : Adaptive Projected Guidance (Sadat et al., "Eliminating Oversaturation and Artifacts of
                     High Guidance Scales in Diffusion Models", ICLR 2025). Applied on the denoised (x0)
                     prediction, as in the paper, then converted back to v for the scheduler.
  * --negative     : negative prompt used as the "unconditional" branch instead of the all-zero embedding.
  * --csv/--col    : alternative (rewritten) prompts.

    python generate_plus.py --out_dir ../gen/apg_w10 --guidance apg --scale 10 --offload
"""

import argparse
import os

import pandas as pd
import soundfile as sf
import torch
from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from cfg import cfg

DIMS = (-1, -2)


class MomentumBuffer:
    def __init__(self, momentum):
        self.momentum, self.avg = momentum, 0

    def update(self, x):
        self.avg = x + self.momentum * self.avg
        return self.avg


def apg(pred_uncond, pred_cond, w, buf=None, eta=0.0, r=0.0):
    diff = pred_cond - pred_uncond
    if buf is not None:
        diff = buf.update(diff)
    if r > 0:
        diff = diff * torch.clamp(r / diff.norm(p=2, dim=DIMS, keepdim=True), max=1.0)
    c = torch.nn.functional.normalize(pred_cond.flatten(1), dim=1).view_as(pred_cond)
    par = (diff * c).sum(dim=DIMS, keepdim=True) * c
    orth = diff - par
    return pred_cond + (w - 1) * (orth + eta * par)


@torch.no_grad()
def generate(pipe, prompt, scale=7.0, steps=100, seconds=10.0, seed=0, guidance="cfg", negative=None,
             eta=0.0, momentum=-0.5, r=0.0, interval=None, concepts=None, concept_scale=3.0):
    """interval=(lo, hi): apply guidance only when lo <= sigma <= hi, else use the conditional output
    (Kynkaanniemi et al., "Applying Guidance in a Limited Interval Improves Sample and Distribution Quality
    in Diffusion Models", NeurIPS 2024).
    concepts=[str,...]: composable guidance (Liu et al., ECCV 2022):
        v = v_u + w (v_c - v_u) + concept_scale / K * sum_k (v_k - v_u)."""
    device = pipe._execution_device
    sr = pipe.vae.config.sampling_rate
    sch = pipe.scheduler

    prompt_embeds = pipe.encode_prompt(prompt, device, do_classifier_free_guidance=False)
    s0, s1 = pipe.encode_duration(0.0, seconds, device, False, batch_size=1)
    cond = torch.cat([prompt_embeds, s0, s1], dim=1)
    if negative:
        neg = pipe.encode_prompt(negative, device, do_classifier_free_guidance=False)
        uncond = torch.cat([neg, s0, s1], dim=1)
    else:
        uncond = torch.zeros_like(cond)
    conds = [uncond, cond]
    for c in concepts or []:
        conds.append(torch.cat([pipe.encode_prompt(c, device, do_classifier_free_guidance=False), s0, s1], dim=1))
    nb = len(conds)
    ehs = torch.cat(conds)
    ghs = torch.cat([s0, s1], dim=2).repeat(nb, 1, 1)

    gen = torch.Generator(device).manual_seed(seed)
    sch.set_timesteps(steps, device=device)
    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, int(pipe.transformer.config.sample_size),
                                   cond.dtype, device, gen, audio_channels=pipe.vae.config.audio_channels)
    extra = pipe.prepare_extra_step_kwargs(gen, 0.0)
    rot = get_1d_rotary_pos_embed(pipe.rotary_embed_dim, latents.shape[2] + ghs.shape[1],
                                  use_real=True, repeat_interleave_real=False)
    buf = MomentumBuffer(momentum) if (guidance == "apg" and momentum != 0) else None

    for i, t in enumerate(pipe.progress_bar(sch.timesteps)):
        inp = sch.scale_model_input(torch.cat([latents] * nb), t)
        v = pipe.transformer(inp, t.unsqueeze(0), encoder_hidden_states=ehs, global_hidden_states=ghs,
                             rotary_embedding=rot, return_dict=False)[0]
        vs = v.chunk(nb)
        v_u, v_c = vs[0], vs[1]
        sigma_now = float(sch.sigmas[i])
        if interval is not None and not (interval[0] <= sigma_now <= interval[1]):
            v = v_c  # outside the guidance interval: plain conditional prediction (w = 1)
        elif guidance == "cfg":
            v = cfg(v_u, v_c, scale)
            if concepts:
                v = v + concept_scale / len(concepts) * sum(vk - v_u for vk in vs[2:])
        else:  # APG in x0 space
            sigma = sch.sigmas[i].to(latents.device)
            sd = sch.config.sigma_data
            c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
            c_out = -sigma * sd / (sigma ** 2 + sd ** 2) ** 0.5
            x0_u = (c_skip * latents + c_out * v_u).float()
            x0_c = (c_skip * latents + c_out * v_c).float()
            x0 = apg(x0_u, x0_c, scale, buf, eta, r)
            v = ((x0 - c_skip * latents.float()) / c_out).to(v_c.dtype)
        latents = sch.step(v, t, latents, **extra).prev_sample

    audio = pipe.vae.decode(latents).sample
    return audio[0, :, : int(seconds * sr)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="test.csv")
    p.add_argument("--col", default="caption")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--guidance", choices=["cfg", "apg"], default="cfg")
    p.add_argument("--scale", type=float, default=7.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=-0.5)
    p.add_argument("--r", type=float, default=0.0, help="APG norm threshold (0 = off)")
    p.add_argument("--negative", default=None)
    p.add_argument("--interval", type=float, nargs=2, default=None, metavar=("SIGMA_LO", "SIGMA_HI"))
    p.add_argument("--concepts", default=None, help="concepts.csv -> composable guidance")
    p.add_argument("--concept_scale", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only first N prompts (quick tests)")
    p.add_argument("--offload", action="store_true")
    a = p.parse_args()

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload() if a.offload else pipe.to("cuda")
    sr = pipe.vae.config.sampling_rate

    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.csv)
    if a.limit:
        df = df.head(a.limit)
    cmap = {}
    if a.concepts:
        for _, c in pd.read_csv(a.concepts).iterrows():
            cmap[c["id"]] = [f"{c['genre']} music", ", ".join(c["instruments"].split(";")),
                             ", ".join(c["mood"].split(";")) + " mood"]
    for _, row in df.iterrows():
        path = os.path.join(a.out_dir, f"{row['id']}.wav")
        if os.path.exists(path):
            continue
        audio = generate(pipe, row[a.col], a.scale, a.steps, a.seconds, a.seed, a.guidance, a.negative,
                         a.eta, a.momentum, a.r, a.interval, cmap.get(row["id"]), a.concept_scale)
        sf.write(path, audio.T.float().cpu().numpy(), sr)
        print(f"[{row['id']}] -> {path}", flush=True)


if __name__ == "__main__":
    main()
