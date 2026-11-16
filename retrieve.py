"""Task 1: for each test caption, retrieve the closest MusicCaps clip with LAION-CLAP.

Uses music_audioset_epoch_15_esc_90.14.pt (same checkpoint as the CLAP metric).

    python retrieve.py --audio_dir ../audio_mp3 --out_dir ../retrieval
"""

import argparse
import os
import shutil

import librosa
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

import laion_clap

CKPT_REPO, CKPT_FILE = "lukewys/laion_clap", "music_audioset_epoch_15_esc_90.14.pt"


def load_clap(device):
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    ckpt_path = hf_hub_download(CKPT_REPO, CKPT_FILE)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    # newer transformers no longer keep this buffer in the state dict
    state.pop("text_branch.embeddings.position_ids", None)
    model.model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def embed_audio_dir(model, files, batch_size=32, sr=48000, seconds=10):
    n = sr * seconds
    embs = []
    for i in range(0, len(files), batch_size):
        batch = []
        for f in files[i:i + batch_size]:
            y, _ = librosa.load(f, sr=sr, mono=True)
            y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
            batch.append(y)
        x = torch.from_numpy(np.stack(batch)).float()
        e = model.get_audio_embedding_from_data(x=x, use_tensor=True)
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        print(f"\raudio {min(i + batch_size, len(files))}/{len(files)}", end="", flush=True)
    print()
    return torch.cat(embs).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", default="../audio_mp3")
    p.add_argument("--test_csv", default="test.csv")
    p.add_argument("--out_dir", default="../retrieval")
    p.add_argument("--cache", default="../musiccaps_clap_emb.npz")
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_clap(device)

    files = sorted(f for f in os.listdir(args.audio_dir) if f.endswith(".mp3"))
    if os.path.exists(args.cache):
        c = np.load(args.cache, allow_pickle=True)
        assert list(c["files"]) == files, "cache is stale, delete it"
        audio_emb = c["emb"]
    else:
        audio_emb = embed_audio_dir(model, [os.path.join(args.audio_dir, f) for f in files])
        np.savez(args.cache, files=np.array(files), emb=audio_emb)

    test = pd.read_csv(args.test_csv)
    with torch.no_grad():
        text_emb = model.get_text_embedding(test["caption"].tolist(), use_tensor=True)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1).cpu().numpy()

    sim = text_emb @ audio_emb.T  # (100, 5367) cosine similarity
    order = np.argsort(-sim, axis=1)[:, : args.topk]

    os.makedirs(os.path.join(args.out_dir, "audio"), exist_ok=True)
    rows = []
    for i, row in test.iterrows():
        top = order[i]
        best = files[top[0]]
        dst = f"{row['id']}_{best}"
        shutil.copy(os.path.join(args.audio_dir, best), os.path.join(args.out_dir, "audio", dst))
        rows.append({
            "id": row["id"],
            "caption": row["caption"],
            "retrieved_ytid": best[:-4],
            "clap_score": float(sim[i, top[0]]),
            **{f"top{k + 1}": f"{files[j][:-4]} ({sim[i, j]:.3f})" for k, j in enumerate(top)},
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, "retrieval.csv"), index=False)
    print(out[["id", "retrieved_ytid", "clap_score"]].to_string(index=False))
    print(f"mean top-1 CLAP score: {out['clap_score'].mean():.4f}")


if __name__ == "__main__":
    main()
