"""MQAScore: concept-level music-text alignment with an audio LLM as judge (inspired by VQAScore / AQAScore).

For every clip, the caption is decomposed into atomic concepts (concepts.csv: genre, instruments, mood).
For each concept we ask Qwen2-Audio-7B-Instruct a yes/no question and take its confidence

    s(c) = P("Yes") / (P("Yes") + P("No"))       (next-token probabilities, case variants summed)

MQAScore(clip) = mean_c s(c);  we also report per-category means and a global question with the full caption.

    python mqascore.py ../gen/cfg10 [more dirs] [--shuffle]
--shuffle pairs each clip with the concepts of a different caption (sanity control: score should drop).
Per-clip scores -> <dir>/mqascore.csv, summary appended to ../eval/mqascore.csv
"""

import argparse
import os

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def questions(caption, row):
    # "strict" template: chosen by mqa_probe.py (best matched-vs-mismatched AUC, least yes-bias)
    sfx = " Answer 'Yes' only if you are confident, otherwise answer 'No'."
    qs = [("genre", f"Is the genre of this music {row['genre']}?")]
    qs += [("instrument", f"Is {inst} clearly audible in this recording?") for inst in row["instruments"].split(";")]
    qs += [("mood", f"Is the mood of this music {m}?") for m in row["mood"].split(";")]
    qs = [(k, q + sfx) for k, q in qs]
    glob = ("global", f'Does this recording match the following description? "{caption}"' + sfx)
    return qs, glob


class Judge:
    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.proc.tokenizer.padding_side = "left"
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL, device_map="cuda", dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16)).eval()
        tok = self.proc.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", "yes", " Yes", " yes"]})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["No", "no", " No", " no"]})
        self.sr = self.proc.feature_extractor.sampling_rate

    def ask(self, audio, qs, chunk=2):
        return np.concatenate([self._ask(audio, qs[k:k + chunk]) for k in range(0, len(qs), chunk)])

    @torch.no_grad()
    def _ask(self, audio, qs):
        texts = []
        for q in qs:
            conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"},
                                                 {"type": "text", "text": q}]}]
            texts.append(self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False))
        inp = self.proc(text=texts, audio=[audio] * len(qs), sampling_rate=self.sr, return_tensors="pt",
                        padding=True).to("cuda")
        inp["input_features"] = inp["input_features"].half()
        logits = self.model(**inp).logits[:, -1].float()
        p = logits.softmax(-1)
        py, pn = p[:, self.yes].sum(-1), p[:, self.no].sum(-1)
        return (py / (py + pn)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    test = pd.read_csv(os.path.join(HERE, "test.csv")).set_index("id")
    con = pd.read_csv(os.path.join(HERE, "concepts.csv")).set_index("id")
    ids = list(test.index)[: a.limit] if a.limit else list(test.index)
    pair = {i: i for i in ids}
    if a.shuffle:
        pair = {i: ids[(k + len(ids) // 2) % len(ids)] for k, i in enumerate(ids)}  # fixed offset: never itself

    judge = Judge()
    out_csv = os.path.join(ROOT, "eval", "mqascore.csv")
    for d in a.dirs:
        d = os.path.abspath(d)
        rows = []
        for i in ids:
            f = os.path.join(d, f"{i}.wav")
            y, _ = librosa.load(f, sr=judge.sr, mono=True)
            j = pair[i]
            qs, glob = questions(test.loc[j, "caption"], con.loc[j])
            s = judge.ask(y, [q for _, q in qs] + [glob[1]])
            kinds = [k for k, _ in qs]
            cs = s[:-1]
            rows.append({"id": i, "mqascore": cs.mean(), "global": s[-1],
                         **{k: cs[[x == k for x in kinds]].mean() for k in ["genre", "instrument", "mood"]}})
            print(f"\r{os.path.basename(d)} {len(rows)}/{len(ids)}", end="", flush=True)
        print()
        per = pd.DataFrame(rows)
        tag = os.path.basename(d) + ("_shuffled" if a.shuffle else "")
        per.to_csv(os.path.join(d, f"mqascore{'_shuffled' if a.shuffle else ''}.csv"), index=False)
        summ = {"run": tag, **per.drop(columns="id").mean().to_dict()}
        print(summ)
        df = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["run"])
        df = pd.concat([df[df["run"] != tag], pd.DataFrame([summ])])
        df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
