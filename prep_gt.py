"""Build av-benchmark GT dir: retrieved MusicCaps clip for caption i -> gt/{i}.wav, plus text csv."""
import os, librosa, pandas as pd, soundfile as sf
r = pd.read_csv("../retrieval/retrieval.csv")
os.makedirs("../eval/gt", exist_ok=True)
for _, row in r.iterrows():
    y, sr = librosa.load(f"../audio_mp3/{row['retrieved_ytid']}.mp3", sr=44100, mono=False)
    sf.write(f"../eval/gt/{row['id']}.wav", y.T if y.ndim == 2 else y, sr)
r[["id", "caption"]].rename(columns={"id": "name"}).to_csv("../eval/text.csv", index=False)
print("done", len(r))
