"""Prompt-design probe for MQAScore: which question template best separates matched vs mismatched concepts?"""
import os

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mqascore import HERE, ROOT, Judge

TEMPLATES = {
    "plain": {"genre": "Is this {} music?", "instrument": "Can you hear {} in this audio?",
              "mood": "Does this music sound {}?", "suffix": " Answer yes or no."},
    "strict": {"genre": "Is the genre of this music {}?", "instrument": "Is {} clearly audible in this recording?",
               "mood": "Is the mood of this music {}?",
               "suffix": " Answer 'Yes' only if you are confident, otherwise answer 'No'."},
    "describe": {"genre": "Listen to the music. Question: is it {}?",
                 "instrument": "Listen to the music. Question: is there a {} playing?",
                 "mood": "Listen to the music. Question: is it {}?", "suffix": " Reply with Yes or No."},
}

N = 12
judge = Judge()
con = pd.read_csv(os.path.join(HERE, "concepts.csv")).set_index("id")
ids = list(con.index)[:N]
other = {i: ids[(k + N // 2) % N] for k, i in enumerate(ids)}
for name, t in TEMPLATES.items():
    lab, sc, cat = [], [], []
    for i in ids:
        y, _ = librosa.load(os.path.join(ROOT, "eval", "gt", f"{i}.wav"), sr=judge.sr, mono=True)
        for j, l in [(i, 1), (other[i], 0)]:
            r = con.loc[j]
            qs = [("genre", r["genre"])] + [("instrument", x) for x in r["instruments"].split(";")] + \
                 [("mood", x) for x in r["mood"].split(";")]
            s = judge.ask(y, [t[k].format(v) + t["suffix"] for k, v in qs])
            lab += [l] * len(s); sc += list(s); cat += [k for k, _ in qs]
    lab, sc, cat = np.array(lab), np.array(sc), np.array(cat)
    msg = f"{name:9s} AUC={roc_auc_score(lab, sc):.3f} match={sc[lab == 1].mean():.3f} mismatch={sc[lab == 0].mean():.3f}"
    for c in ["genre", "instrument", "mood"]:
        m = cat == c
        msg += f" | {c} AUC={roc_auc_score(lab[m], sc[m]):.2f}"
    print(msg, flush=True)
