"""Aggregate all results for the report: tables (markdown), bootstrap CIs, figures.

    python analyze.py      -> ../report/tables.md, ../report/figs/*.png
"""

import os

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report")
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)

# reference palette (light mode), fixed order
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200})

METRICS = ["FD-VGG", "FD-PANN", "FD-PASST", "KL-PANNS-softmax", "KL-PASST-softmax", "LAION-CLAP-Score"]
SHORT = {"FD-VGG": "FD-VGG↓", "FD-PANN": "FD-PANN↓", "FD-PASST": "FD-PaSST↓", "KL-PANNS-softmax": "KL-PANNs↓",
         "KL-PASST-softmax": "KL-PaSST↓", "LAION-CLAP-Score": "CLAP↑"}


def per_clip_clap(run):
    t = torch.load(os.path.join(ROOT, "eval", "gt", "cache", "clap_laion_text.pth"), weights_only=True)
    a = torch.load(os.path.join(ROOT, "gen", run, "cache", "clap_laion_audio.pth"), weights_only=True)
    keys = sorted(a, key=int)
    return pd.Series([torch.cosine_similarity(t[k], a[k], dim=0).item() for k in keys], index=[int(k) for k in keys])


def paired_boot(x, y, n=10000, seed=0):
    """95% CI of mean(x - y) over clips (paired bootstrap)."""
    d = (x - y).values
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def md(df, floatfmt=3):
    return df.to_markdown(index=False, floatfmt=f".{floatfmt}f")


def main():
    res = pd.read_csv(os.path.join(ROOT, "eval", "results.csv"))
    res = res[~res["run"].str.startswith("_")].set_index("run")
    lines = []

    # ---------- retrieval
    r = pd.read_csv(os.path.join(ROOT, "retrieval", "retrieval.csv"))
    lines += ["## Retrieval", f"mean top-1 CLAP = {r['clap_score'].mean():.4f}, "
              f"min = {r['clap_score'].min():.3f}, max = {r['clap_score'].max():.3f}, "
              f"unique clips = {r['retrieved_ytid'].nunique()}/100", ""]

    # ---------- main table
    order = [x for x in ["cfg4", "cfg7", "cfg10", "cfg13", "apg10", "apg15", "interval10", "compose7_3",
                         "pe_cfg7", "neg_cfg7", "cfg10_50steps", "cfg7_seed1", "cfg10_seed1",
                         "interval10_seed1", "compose7_3_seed1"] if x in res.index]
    t = res.loc[order, METRICS].rename(columns=SHORT).reset_index()
    lines += ["## All runs (av-benchmark)", md(t), ""]

    # ---------- seed-averaged
    pairs = {"cfg7": "cfg7_seed1", "cfg10": "cfg10_seed1", "interval10": "interval10_seed1",
             "compose7_3": "compose7_3_seed1"}
    rows = []
    for a, b in pairs.items():
        if a in res.index and b in res.index:
            m = res.loc[[a, b], METRICS]
            rows.append({"run": a, **{SHORT[k]: f"{m[k].mean():.3f} ± {abs(m[k].iloc[0] - m[k].iloc[1]) / 2:.3f}"
                                      for k in METRICS}})
    if rows:
        lines += ["## Two-seed mean ± half-range", md(pd.DataFrame(rows)), ""]

    # ---------- per-clip CLAP paired bootstrap vs baseline (same seed)
    rows = []
    for run in order:
        base = "cfg7_seed1" if run.endswith("seed1") else "cfg7"
        if run == base:
            continue
        try:
            mean, lo, hi = paired_boot(per_clip_clap(run), per_clip_clap(base))
        except FileNotFoundError:
            continue
        rows.append({"run": run, "vs": base, "ΔCLAP": mean, "95% CI low": lo, "95% CI high": hi,
                     "significant": "yes" if lo > 0 or hi < 0 else "no"})
    lines += ["## Per-clip CLAP difference vs CFG w=7 (paired bootstrap, 10k)", md(pd.DataFrame(rows), 4), ""]

    # ---------- MQAScore
    mq_path = os.path.join(ROOT, "eval", "mqascore.csv")
    if os.path.exists(mq_path):
        mq = pd.read_csv(mq_path).set_index("run")
        lines += ["## MQAScore (Qwen2-Audio-7B-Instruct judge)", md(mq.reset_index()), ""]
        rows = []
        for run in mq.index:
            if run in ("gt", "cfg7", "cfg7_seed1") or run.endswith("_shuffled"):
                continue
            base = "cfg7_seed1" if run.endswith("seed1") else "cfg7"
            f = lambda x: pd.read_csv(os.path.join(ROOT, "eval" if x == "gt" else "gen", x, "mqascore.csv"))\
                .set_index("id")["mqascore"]  # noqa: E731
            try:
                mean, lo, hi = paired_boot(f(run), f(base))
            except FileNotFoundError:
                continue
            rows.append({"run": run, "vs": base, "ΔMQA": mean, "95% CI low": lo, "95% CI high": hi,
                         "significant": "yes" if lo > 0 or hi < 0 else "no"})
        if rows:
            lines += ["## Per-clip MQAScore difference vs CFG w=7 (paired bootstrap)", md(pd.DataFrame(rows), 4), ""]

    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))

    # ---------- Fig 1: guidance-scale sweep (small multiples, one axis each)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, k in zip(axs, ["FD-VGG", "KL-PASST-softmax", "LAION-CLAP-Score"]):
        ws = [w for w in [4, 7, 10, 13] if f"cfg{w}" in res.index]
        ax.plot(ws, [res.loc[f"cfg{w}", k] for w in ws], "-o", color=C1, lw=2, ms=5, label="CFG")
        wa = [w for w in [10, 15] if f"apg{w}" in res.index]
        ax.plot(wa, [res.loc[f"apg{w}", k] for w in wa], "-s", color=C2, lw=2, ms=5, label="APG")
        ws1 = [w for w in [7, 10] if f"cfg{w}_seed1" in res.index]
        ax.plot(ws1, [res.loc[f"cfg{w}_seed1", k] for w in ws1], "--o", color=C1, mfc="none", mec=C1, lw=1,
                ms=6, label="CFG (seed 1)")
        ax.set_title(SHORT[k], color=INK, fontsize=9)
        ax.set_xlabel("guidance scale w")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scale_sweep.png"))

    # ---------- Fig 2: methods vs baseline, both seeds (dot plot)
    meths = [("cfg7", "CFG w=7 (baseline)"), ("cfg10", "CFG w=10"), ("interval10", "Guidance interval w=10"),
             ("compose7_3", "Composable 7+3")]
    meths = [m for m in meths if m[0] in res.index]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for ax, k in zip(axs, ["FD-VGG", "KL-PASST-softmax", "LAION-CLAP-Score"]):
        for yi, (m, _) in enumerate(meths):
            v0 = res.loc[m, k]
            ax.plot([v0], [yi], "o", color=C1, ms=6)
            if f"{m}_seed1" in res.index:
                v1 = res.loc[f"{m}_seed1", k]
                ax.plot([v0, v1], [yi, yi], color=GRID, lw=2, zorder=0)
                ax.plot([v1], [yi], "o", color=C2, ms=6)
        ax.set_title(SHORT[k], color=INK, fontsize=9)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(meths)), [n for _, n in meths])
    axs[0].invert_yaxis()
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], ls="", marker="o", color=C1), Line2D([], [], ls="", marker="o", color=C2)],
               ["seed 0", "seed 1"], frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIG, "methods_seeds.png"))

    # ---------- Fig 3: MQAScore by concept category
    if os.path.exists(mq_path):
        mq = pd.read_csv(mq_path).set_index("run")
        show = [("gt", "MusicCaps\n(retrieved)"), ("gt_shuffled", "MusicCaps\nshuffled captions"),
                ("cfg7", "CFG w=7"), ("cfg10", "CFG w=10"), ("compose7_3", "Composable\n7+3"),
                ("cfg10_shuffled", "CFG w=10\nshuffled captions")]
        show = [s for s in show if s[0] in mq.index]
        cats = [("genre", C1), ("instrument", C2), ("mood", C3)]
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        bw = 0.26
        for ci, (c, col) in enumerate(cats):
            xs = np.arange(len(show)) + (ci - 1) * (bw + 0.02)
            ax.bar(xs, [mq.loc[s, c] for s, _ in show], width=bw, color=col, label=c, edgecolor=SURF, lw=1)
        ax.set_xticks(range(len(show)), [n for _, n in show], fontsize=7)
        ax.set_ylabel("P(Yes) / (P(Yes)+P(No))")
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "mqascore.png"))


if __name__ == "__main__":
    main()
