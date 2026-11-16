"""Extra analyses (no GPU needed):
  (7) agreement between CLAP and MQAScore, per clip and per system
  (8) retrieval failure cases (needs ../musiccaps-public.csv from the google/MusicCaps HF dataset)

    python analyze_extra.py   -> ../report/extra.md, ../report/figs/clap_vs_mqa.png
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from analyze import C1, C2, FIG, INK, INK2, OUT, ROOT, md, per_clip_clap, plt

RUNS = ["cfg4", "cfg7", "cfg10", "cfg13", "apg10", "apg15", "interval10", "compose7_3", "pe_cfg7", "neg_cfg7",
        "cfg10_50steps", "cfg7_seed1", "cfg10_seed1", "interval10_seed1", "compose7_3_seed1"]


def boot_rho(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    rs = [spearmanr(x[i], y[i])[0] for i in rng.integers(0, len(x), (n, len(x)))]
    return np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def mqa(run):
    return pd.read_csv(os.path.join(ROOT, "gen", run, "mqascore.csv")).set_index("id")


def part7(lines):
    ret = pd.read_csv(os.path.join(ROOT, "retrieval", "retrieval.csv")).set_index("id")
    gt_mq = pd.read_csv(os.path.join(ROOT, "eval", "gt", "mqascore.csv")).set_index("id")

    # --- per clip, per run
    rows, pooled = [], []
    for run in RUNS:
        c, m = per_clip_clap(run), mqa(run)
        df = pd.DataFrame({"clap": c, **{k: m[k] for k in ["mqascore", "global", "genre", "instrument", "mood"]}})
        df["run"] = run
        pooled.append(df)
        rows.append({"audio": run, **{f"ρ({k})": spearmanr(df["clap"], df[k])[0]
                                      for k in ["mqascore", "global", "genre", "instrument", "mood"]}})
    df = pd.DataFrame({"clap": ret["clap_score"], **{k: gt_mq[k] for k in
                                                       ["mqascore", "global", "genre", "instrument", "mood"]}})
    rows.insert(0, {"audio": "MusicCaps reference (CLAP = retrieval score)",
                    **{f"ρ({k})": spearmanr(df["clap"], df[k])[0]
                       for k in ["mqascore", "global", "genre", "instrument", "mood"]}})
    per = pd.DataFrame(rows)
    allc = pd.concat(pooled)
    lo, hi = boot_rho(allc["clap"], allc["mqascore"])
    rho_pooled = spearmanr(allc["clap"], allc["mqascore"])[0]

    # within-caption: does the run that CLAP prefers for a caption also get the higher MQAScore?
    piv_c = allc.pivot_table(index=allc.index, columns="run", values="clap")
    piv_m = allc.pivot_table(index=allc.index, columns="run", values="mqascore")
    within = np.nanmean([spearmanr(piv_c.loc[i], piv_m.loc[i])[0] for i in piv_c.index])

    # caption difficulty: mean over runs, per caption
    diff_c, diff_m = piv_c.mean(1), piv_m.mean(1)
    rho_diff = spearmanr(diff_c, diff_m)[0]
    rho_ret_gen = spearmanr(ret["clap_score"], diff_c)[0]

    # --- system level (15 runs)
    sysm = pd.DataFrame({"clap": [per_clip_clap(r).mean() for r in RUNS], "mqa": [mqa(r)["mqascore"].mean() for r in RUNS]},
                        index=RUNS)
    rs, ps = spearmanr(sysm["clap"], sysm["mqa"])
    kt, pk = kendalltau(sysm["clap"], sysm["mqa"])
    no_neg = sysm.drop("neg_cfg7")
    rs2, ps2 = spearmanr(no_neg["clap"], no_neg["mqa"])

    lines += [
        "## (7) Agreement between CLAP and MQAScore", "",
        "Spearman ρ between per-clip LAION-CLAP and per-clip MQAScore (and its parts), 100 clips per row:", "",
        md(per, 3), "",
        f"* pooled over all 15 generated runs (1500 clips): ρ = {rho_pooled:.3f} (95% bootstrap CI [{lo:.3f}, {hi:.3f}])",
        f"* caption difficulty (per-caption mean over the 15 runs): ρ(CLAP, MQAScore) = {rho_diff:.3f}",
        f"* within a caption, across the 15 runs: mean ρ = {within:.3f} (do both metrics prefer the same run?)",
        f"* retrieval score vs mean generated CLAP of the same caption: ρ = {rho_ret_gen:.3f}",
        f"* system level (15 runs): Spearman ρ = {rs:.3f} (p = {ps:.3f}), Kendall τ = {kt:.3f} (p = {pk:.3f}); "
        f"without neg_cfg7: ρ = {rs2:.3f} (p = {ps2:.3f})", ""]

    # --- figure: per-caption difficulty scatter + system level scatter
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axs[0]
    ax.scatter(diff_c, diff_m, s=14, color=C1, alpha=0.8, edgecolor="white", lw=0.5)
    ax.set_xlabel("CLAP (mean over 15 runs)")
    ax.set_ylabel("MQAScore (mean over 15 runs)")
    ax.set_title(f"Per caption (n=100), ρ = {rho_diff:.2f}", color=INK, fontsize=9)
    ax = axs[1]
    ax.scatter(sysm["clap"], sysm["mqa"], s=30, color=C2, edgecolor="white", lw=0.8, zorder=3)
    for r in ["neg_cfg7", "cfg4", "cfg10", "compose7_3_seed1", "interval10", "pe_cfg7"]:
        ax.annotate(r, (sysm.loc[r, "clap"], sysm.loc[r, "mqa"]), fontsize=6.5, color=INK2,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("CLAP (run mean)")
    ax.set_ylabel("MQAScore (run mean)")
    ax.set_title(f"Per run (n=15), ρ = {rs:.2f}", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "clap_vs_mqa.png"))
    return diff_c, diff_m


def part8(lines, diff_c, diff_m):
    ret = pd.read_csv(os.path.join(ROOT, "retrieval", "retrieval.csv")).set_index("id")
    mc = pd.read_csv(os.path.join(ROOT, "musiccaps-public.csv")).drop_duplicates("ytid").set_index("ytid")
    gt_mq = pd.read_csv(os.path.join(ROOT, "eval", "gt", "mqascore.csv")).set_index("id")
    con = pd.read_csv(os.path.join(HERE := os.path.dirname(os.path.abspath(__file__)), "concepts.csv")).set_index("id")

    short = lambda s, n=150: s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"  # noqa: E731
    lines += ["## (8) Retrieval failure cases", ""]

    # score distribution & relation to judge
    rho, p = spearmanr(ret["clap_score"], gt_mq["mqascore"])
    q = ret["clap_score"].quantile([.1, .5, .9])
    lines += [f"Top-1 CLAP score: 10th pct {q[.1]:.3f}, median {q[.5]:.3f}, 90th pct {q[.9]:.3f}. "
              f"Retrieval score vs MQAScore of the retrieved clip (judge asks about the *test* caption's concepts): "
              f"ρ = {rho:.3f} (p = {p:.3f}).", ""]

    low = ret.sort_values("clap_score").head(6)
    rows = []
    for i, r in low.iterrows():
        m = mc.loc[r["retrieved_ytid"]] if r["retrieved_ytid"] in mc.index else None
        rows.append({"id": i, "score": f"{r['clap_score']:.3f}", "MQA(ref)": f"{gt_mq.loc[i, 'mqascore']:.2f}",
                     "test caption": short(r["caption"]),
                     "retrieved MusicCaps caption": short(m["caption"]) if m is not None else "?",
                     "concepts (genre / instr. / mood)": f"{con.loc[i, 'genre']} / {con.loc[i, 'instruments']} / {con.loc[i, 'mood']}"})
    lines += ["### Six lowest-scoring retrievals", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # biggest judge failures: concepts with lowest per-clip MQA on the reference
    worst = gt_mq.sort_values("mqascore").head(5)
    rows = []
    for i, r in worst.iterrows():
        y = ret.loc[i, "retrieved_ytid"]
        rows.append({"id": i, "retr. score": f"{ret.loc[i, 'clap_score']:.3f}", "MQA(ref)": f"{r['mqascore']:.2f}",
                     "genre / instr. / mood": f"{r['genre']:.2f} / {r['instrument']:.2f} / {r['mood']:.2f}",
                     "test concepts": f"{con.loc[i, 'genre']} / {con.loc[i, 'instruments']} / {con.loc[i, 'mood']}",
                     "retrieved aspects": short(", ".join(eval(mc.loc[y, "aspect_list"])), 120) if y in mc.index else "?"})
    lines += ["### Five references the MQAScore judge rates lowest", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # hub clips: retrieved for many different captions
    vc = ret["retrieved_ytid"].value_counts()
    hubs = vc[vc > 1]
    rows = []
    for y, n in hubs.head(5).items():
        ids = list(ret.index[ret["retrieved_ytid"] == y])
        rows.append({"ytid": y, "times retrieved": n, "test ids": ", ".join(map(str, ids)),
                     "mean score": f"{ret.loc[ids, 'clap_score'].mean():.3f}",
                     "MusicCaps caption": short(mc.loc[y, "caption"], 120) if y in mc.index else "?"})
    lines += [f"### Hub clips: {len(hubs)} MusicCaps clips are top-1 for more than one caption "
              f"({int(hubs.sum())} captions in total; 69 distinct clips for 100 captions)", "",
              pd.DataFrame(rows).to_markdown(index=False), ""]

    # genres poorly served by the corpus
    g = pd.DataFrame({"genre": con["genre"], "score": ret["clap_score"]}).groupby("genre")["score"].agg(["mean", "count"])
    g = g[g["count"] >= 2].sort_values("mean")
    lines += ["### Mean retrieval score by test genre (genres with ≥ 2 captions)", "",
              g.reset_index().to_markdown(index=False, floatfmt=".3f"), ""]


def main():
    lines = []
    diff_c, diff_m = part7(lines)
    part8(lines, diff_c, diff_m)
    with open(os.path.join(OUT, "extra.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
