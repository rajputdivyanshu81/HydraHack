"""Headline chart: held-out Hits@1 per round for DesirePath vs flat baselines, plus ablations.

    python -m eval.plot [--suffix _offline]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
STYLE = {  # name -> (label, color, linestyle, linewidth)
    "desirepath": ("DesirePath (heal + plan learning)", "#cf4a1b", "-", 3.0),
    "heal_only": ("ablation: healing only", "#e3954f", "--", 1.6),
    "plans_only": ("ablation: plan learning only", "#9a6fb0", "--", 1.6),
    "random_heal": ("control: random ingestion, same budget", "#8c959f", ":", 1.8),
    "noisy_fb": ("robustness: 20% wrong feedback", "#d4a72c", "-.", 1.6),
    "no_merge": ("ablation: no entity merge", "#bf8700", ":", 1.4),
}
FLAT = {  # name -> (label, color)
    "full_kb": ("upper bound: complete graph", "#2da44e"),
    "static": ("static damaged graph", "#57606a"),
    "baseline_vector": ("vector store RAG (top-5, all text)", "#0969da"),
    "baseline_llm": ("Claude alone, no retrieval", "#8250df"),
}


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=os.path.join(RES, "curve.png"))
    args = ap.parse_args()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    max_round = 1
    summary = {}
    for name, (label, color, ls, lw) in STYLE.items():
        r = load(name + args.suffix)
        if not r:
            continue
        xs = [x["round"] for x in r["rounds"]]
        ys = [100 * x["heldout"]["hits1"] for x in r["rounds"]]
        max_round = max(max_round, xs[-1])
        ax.plot(xs, ys, ls=ls, lw=lw, color=color, marker="o" if name == "desirepath" else None, label=label)
        if name == "desirepath":
            lo = [100 * x["heldout"]["hits1_ci"][0] for x in r["rounds"]]
            hi = [100 * x["heldout"]["hits1_ci"][1] for x in r["rounds"]]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12, lw=0)
            ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 4),
                        color=color, fontweight="bold")
            if r.get("serve_repair"):
                yr = 100 * r["serve_repair"]["hits1"]
                ax.scatter([xs[-1]], [yr], marker="*", s=220, color=color, zorder=5,
                           label="DesirePath serving with query-time repair")
                ax.annotate(f"{yr:.1f}%", (xs[-1], yr), textcoords="offset points", xytext=(8, -3),
                            color=color, fontweight="bold")
                summary["desirepath_serve_repair"] = yr
            tok = [x["heldout"]["tokens"] for x in r["rounds"]]
            mem = [100 * x["heldout"]["from_memory"] for x in r["rounds"]]
            ax2.bar(xs, tok, color="#cf4a1b", alpha=0.85, label="Claude tokens per held-out query")
            ax2.set_ylim(0, max(tok) * 1.15 if max(tok) > 0 else 1)
            ax2.set_xticks(xs)
            ax2b = ax2.twinx()
            ax2b.plot(xs, mem, color=INK, marker="s", lw=1.8, label="% answered from learned plan memory")
            ax2b.set_ylim(0, 100)
            ax2b.set_ylabel("% from plan memory", color=INK)
            ax2.set_ylabel("tokens / query")
            ax2.set_xlabel("learning round (150 stream questions each)")
            ax2.set_title("Cost falls as the graph learns", loc="left", fontsize=12, color=INK)
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax2b.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=False)
        summary[name] = ys
    for name, (label, color) in FLAT.items():
        r = load(name + (args.suffix if not name.startswith("baseline") else ""))
        if not r:
            continue
        y = 100 * (r["heldout"]["hits1"] if "heldout" in r and "hits1" in r["heldout"] else r["rounds"][0]["heldout"]["hits1"])
        ax.axhline(y, color=color, lw=1.4, ls=(0, (4, 2)) if name != "full_kb" else "-", alpha=0.9)
        ax.text(max_round + 0.05, y, f" {label}  {y:.1f}%", color=color, va="center", fontsize=8.5)
        summary[name] = y
    ax.set_xlim(-0.2, max_round + 0.2)
    ax.set_xticks(range(0, max_round + 1))
    ax.set_ylim(0, 100)
    ax.set_xlabel("learning round (150 stream questions each; held-out set is never learned from)", fontsize=9)
    ax.set_ylabel("held-out Hits@1 (%)")
    ax.set_title("Held-out accuracy rises as the graph repairs itself", loc="left", fontsize=13, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "MetaQA 2/3-hop, 50% of knowledge-graph edges deleted (seed 0); 300 held-out questions, "
                         "95% bootstrap CI shaded. Graph: HydraDB 0.4.0.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.86, 1))
    fig.savefig(args.out, dpi=160)
    json.dump(summary, open(os.path.join(RES, "curve_summary.json"), "w"), indent=1)
    print("wrote", args.out)
    print(json.dumps(summary, indent=1))
    if not args.suffix:
        write_readme_table()


def _fmt(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def write_readme_table():
    """Replace the RESULTS block in README.md with numbers read from results/*.json."""
    rows = []
    main = load("desirepath")
    order = [("baseline_vector", "Vector-store RAG: all 18k passages, top-5, Claude answers"),
             ("baseline_llm", "Claude alone, no retrieval"),
             ("static", "Damaged graph, no learning"),
             ("random_heal", "Random ingestion, same reading budget (control)"),
             ("heal_only", "Healing only"), ("plans_only", "Plan learning only"),
             ("no_merge", "All mechanisms except entity merge"),
             ("noisy_fb", "DesirePath with 20% wrong feedback"),
             ("desirepath", "**DesirePath** (all four mechanisms)"),
             ("full_kb", "Upper bound: complete graph")]
    for name, label in order:
        r = load(name)
        if not r:
            continue
        if "rounds" in r:
            last = r["rounds"][-1]["heldout"]
            rows.append(f"| {label} | {_fmt(last['hits1'])} | {_fmt(last.get('hits1_seen-topic'))} | "
                        f"{_fmt(last.get('hits1_unseen-topic'))} | {last['tokens']:.0f} |")
        else:
            h = r["heldout"]
            rows.append(f"| {label} | {_fmt(h['hits1'])} | {_fmt(h.get('hits1_seen-topic'))} | "
                        f"{_fmt(h.get('hits1_unseen-topic'))} | {h['tokens']:.0f} |")
    if main and main.get("serve_repair"):
        s = main["serve_repair"]
        rows.append(f"| **DesirePath, serving with query-time repair** | {_fmt(s['hits1'])} | "
                    f"{_fmt(s.get('hits1_seen-topic'))} | {_fmt(s.get('hits1_unseen-topic'))} | {s['tokens']:.0f} |")
    lines = ["| System | Held-out Hits@1 | Seen topics | Never-seen topics | Claude tokens / query |",
             "|---|---|---|---|---|"] + rows
    if main:
        r0, rN = main["rounds"][0]["heldout"], main["rounds"][-1]["heldout"]
        lines += ["", f"DesirePath went from {_fmt(r0['hits1'])} to {_fmt(rN['hits1'])} held-out Hits@1 over "
                      f"{main['rounds'][-1]['round']} rounds, while Claude tokens per query fell from "
                      f"{r0['tokens']:.0f} to {rN['tokens']:.0f}."]
        g = main["rounds"][-1]["graph"]
        lines.append(f"Graph edits: {g['text_edges']:,} edges healed from {g['docs_read']:,} passages; "
                     f"{g['plans']} plans learned across {g['templates']} question wordings.")
        if main.get("resolver"):
            rs = main["resolver"]
            lines.append(f"Entity resolution: {rs['normalised']} normalised links, {rs['alias_merge']} alias merges, "
                         f"{rs['new_entity']} new entities, {rs['dropped']} names dropped.")
        if main.get("compressor"):
            c = main["compressor"]
            lines.append(f"Path compression: {c['compressed']} hot traversals compressed into {c['edges']} SHORTCUT "
                         f"edges; {c['invalidated']} invalidated by healing.")
    lt = os.path.join(RES, "load_test_32.json")
    if os.path.exists(lt):
        L = json.load(open(lt))
        lines.append(f"Under load: {L['readers']} concurrent readers plus a live writer: {L['qps']} queries/s, "
                     f"p50 {L['p50_ms']} ms, p95 {L['p95_ms']} ms, {L['errors']} errors.")
        c = L.get("compression") or {}
        if c.get("shortcuts"):
            lines.append(f"SHORTCUT reads: p50 {c['shortcut_p50_ms']} ms vs {c['full_traversal_p50_ms']} ms for the "
                         f"{c['hops_mean']}-hop traversal they replace ({c['speedup']}x).")
    readme = os.path.join(ROOT, "README.md")
    text = open(readme, encoding="utf-8").read()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = head + start + "\n" + "\n".join(lines) + "\n" + end + tail
        open(readme, "w", encoding="utf-8").write(text)
        print("updated README results")


if __name__ == "__main__":
    main()
