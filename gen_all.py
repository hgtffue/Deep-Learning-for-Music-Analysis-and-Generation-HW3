"""Task 2: generate one 10 s clip per test caption with Stable Audio Open + our CFG.

    python gen_all.py --out_dir ../gen/cfg7 --scale 7
    python gen_all.py --out_dir ../gen/pe_cfg7 --csv prompts_rewritten.csv   # prompt-engineered captions
"""

import argparse
import os

import pandas as pd
import soundfile as sf
import torch
from diffusers import StableAudioPipeline

from generate import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="test.csv")
    p.add_argument("--col", default="caption")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scale", type=float, default=7.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offload", action="store_true", help="use model CPU offload (GPUs with < 10 GB)")
    args = p.parse_args()

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    sr = pipe.vae.config.sampling_rate

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    for _, row in df.iterrows():
        path = os.path.join(args.out_dir, f"{row['id']}.wav")
        if os.path.exists(path):
            continue
        audio = generate(pipe, row[args.col], args.scale, args.steps, args.seconds, args.seed)
        sf.write(path, audio.T.float().cpu().numpy(), sr)
        print(f"[{row['id']}] -> {path}", flush=True)


if __name__ == "__main__":
    main()
