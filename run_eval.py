"""Task 3: evaluate a folder of generated clips with av-benchmark against the retrieved MusicCaps clips.

    python run_eval.py ../gen/cfg7 [more dirs ...]

Reference = eval/gt/{id}.wav (CLAP top-1 MusicCaps clip for caption id), so KL is paired per caption.
CLAP score = cosine(text of caption id, generated audio id), music_audioset_epoch_15_esc_90.14.pt.
Results are appended to eval/results.csv.
"""

import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AVB = os.path.join(ROOT, "av-benchmark")
GT = os.path.join(ROOT, "eval", "gt")
KEYS = ["FD-VGG", "FD-PANN", "FD-PASST", "KL-PANNS-softmax", "KL-PASST-softmax", "LAION-CLAP-Score",
        "MS-CLAP-Score", "ISC-PANNS-mean", "ISC-PASST-mean"]


def main():
    out_csv = os.path.join(ROOT, "eval", "results.csv")
    for pred in sys.argv[1:]:
        pred = os.path.abspath(pred)
        cache = os.path.join(pred, "cache")
        cmd = [sys.executable, "evaluate.py", "--gt_audio", GT, "--gt_cache", os.path.join(GT, "cache"),
               "--pred_audio", pred, "--pred_cache", cache, "--audio_length", "10",
               "--skip_video_related", "--num_workers", "0", "--pred_batch_size", "4"]
        subprocess.run(cmd, cwd=AVB, check=True)
        with open(os.path.join(cache, "output_metrics.json")) as f:
            m = json.load(f)
        row = {"run": os.path.basename(pred), **{k: m.get(k) for k in KEYS}}
        print(row)
        df = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
        df = df[df.get("run", pd.Series(dtype=str)) != row["run"]] if len(df) else df
        pd.concat([df, pd.DataFrame([row])]).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
