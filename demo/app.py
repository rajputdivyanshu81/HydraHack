"""DesirePath live demo (Streamlit).

    .venv\\Scripts\\streamlit run demo/app.py

Ask a question -> see the relation plan, the HydraDB traversal, the detected gaps.
Give feedback -> watch the graph heal (text edges written) and the plan memory reweight, then ask again.
"""
from __future__ import annotations

import json
import os
import sys

import streamlit as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from desirepath import llm  # noqa: E402
from desirepath.data import OUT  # noqa: E402
from desirepath.db import Hydra  # noqa: E402
from desirepath.graph import Graph, load_prepared  # noqa: E402
from desirepath.heal import Corpus  # noqa: E402
from desirepath.system import Config, System  # noqa: E402

st.set_page_config(page_title="DesirePath", layout="wide")


@st.cache_resource
def boot():
    db = Hydra()
    ents, kb = load_prepared()
    g = Graph(db, ents)
    g.mirror(kb["kept"])
    offline = not os.getenv("ANTHROPIC_API_KEY")
    cfg = Config(heal="demand", plans="learn", planner="rules" if offline else "llm",
                 extractor="rules" if offline else "llm")
    sysm = System(db, g, Corpus(), cfg, os.path.join(ROOT, "results", "demo.edits.jsonl"))
    wl = json.load(open(os.path.join(OUT, "workload.json")))
    return db, g, sysm, wl, offline


db, g, sysm, wl, offline = boot()

st.title("DesirePath")
st.caption("A self-healing knowledge graph on HydraDB: it repairs itself where users ask. "
           + ("Offline mode (rule planner, regex extractor)." if offline else "Claude Haiku 5.5 planner and extractor."))

left, right = st.columns([1.3, 1])
with left:
    pool = wl["heldout"][:150]
    labels = [f"{q['question']}  ({q['hop']}-hop)" for q in pool]
    i = st.selectbox("Question", range(len(pool)), format_func=lambda k: labels[k])
    q = pool[i]
    repair = st.toggle("Query-time repair (heal from text on a gap, then answer)", value=False)
    if st.button("Ask", type="primary"):
        st.session_state["out"] = sysm.answer_with_repair(q) if repair else sysm.answer(q, probe=True)
        st.session_state["q"] = q
    out = st.session_state.get("out")
    if out and st.session_state.get("q", {}).get("qid") == q["qid"]:
        gold = set(q["answers"])
        top = [(g.name[x], c) for x, c in out["_res"][-1].ranked[:8]] if out["_res"] else []
        via = "SHORTCUT edges (compressed path)" if out.get("shortcut") else out["plan_source"]
        st.markdown(f"**Plan** `{out['plan']}`  · from **{via}** · "
                    f"{out['queries']} HydraDB queries · {out['ms']:.0f} ms · {out['tokens']} Claude tokens"
                    + (" · 🩹 repaired at query time" if out.get("repaired") else ""))
        st.markdown("**Answer:** " + (f"{'✅' if out['hit'] else '❌'} {out['pred']}" if out["pred"] else "❌ nothing found"))
        st.table([{"answer": n, "paths": c, "correct": "✓" if n in gold else ""} for n, c in top])
        gaps = {lvl: [g.name[x] for x in ids][:6] for r in out["_res"] for lvl, ids in r.gaps.items()}
        if gaps:
            st.warning("Gaps detected (nodes missing the next relation): "
                       + "; ".join(f"hop {k}: {', '.join(v)}" for k, v in gaps.items()))
        st.caption("Correct answers: " + ", ".join(sorted(gold)[:10]))
        if st.button("Give feedback and learn"):
            before = sysm.healer.stats["edges_added"]
            sysm.learn(q, out)
            st.success(f"Healed: +{sysm.healer.stats['edges_added'] - before} edges written to HydraDB. "
                       "Plan memory reweighted. Ask again.")
            st.session_state.pop("out")

with right:
    chart = os.path.join(ROOT, "results", "curve.png")
    if os.path.exists(chart):
        st.image(chart, caption="Held-out accuracy by learning round")
    st.subheader("Graph edit log")
    path = sysm.log.path
    if os.path.exists(path):
        lines = open(path, encoding="utf-8").read().splitlines()[-12:]
        for line in reversed(lines):
            e = json.loads(line)
            k = e["kind"]
            if k == "ingest":
                st.markdown(f"🩹 **ingest** {e['edges_added']} edges from {e['docs']} passages "
                            f"about {', '.join(e['entities'][:3])}")
            elif k == "reweight":
                st.markdown(f"⚖️ **reweight** `{e['plan']}` {e['cost_before']} → {e['cost_after']}  \n"
                            f"<small>{e['template']}</small>", unsafe_allow_html=True)
            elif k == "merge":
                st.markdown(f"🔗 **merge** “{e['alias']}” → {e['into']}")
            elif k == "new_entity":
                st.markdown(f"➕ **new entity** {e['name']} ({e.get('entity_kind', '')})")
            elif k == "compress":
                st.markdown(f"⚡ **compress** {e['hops']}-hop `{e['plan']}` → {e['shortcut_edges']} SHORTCUT edges")
            elif k == "decompress":
                st.markdown(f"♻️ **invalidate shortcut** `{e['plan']}` ({e['cause']})")
    st.caption(f"Claude usage this session: {llm.USAGE.snapshot()}")
