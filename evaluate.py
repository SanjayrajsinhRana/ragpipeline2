#!/usr/bin/env python3
"""
Validation for the GEP June 2026 pipeline.

Retrieval (no API calls): a section from the expected chapter / region / box that
contains the expected fact must be among the 3 passages sent to the model (PASS), and
no passage may come from outside the named chapter. This is the
"right section, not a page that shares a few words" bar. Reported as top-1 hit and MRR.
Run with --ablate to also score the same questions with the query filters switched
off, which shows what the metadata routing buys.

Generation (Gemini, free tier): the answer must contain the expected facts, must not
contain figures that belong to another region / chapter, and the out-of-scope
question must be refused.

    python evaluate.py              # retrieval + generation
    python evaluate.py --retrieval-only [--ablate]
"""

import argparse
import re
import sys

from src.generation import REFUSAL
from src.pipeline import GEPPipeline
from src.retrieval import retrieve
from src.query import rule_plan

OTHER_REGION_FIGURES = [r"6\.3 percent", r"(?<![\d.])4\.2 percent", r"1\.6 percent", r"(?<![\d.])2\.1 percent", r"(?<![\d.])2\.2 percent"]


def forbid_except(*keep):
    return [p for p in OTHER_REGION_FIGURES if p not in keep]


CASES = [
    # --- the three required questions -------------------------------------
    {"id": "Q1", "q": "What is the growth outlook for South Asia in this report?",
     "chapter": "ch2", "region": "South Asia", "facts": [r"6\.3 percent"],
     "forbid": forbid_except(r"6\.3 percent")},
    {"id": "Q2", "q": "What does Chapter 3 say about sovereign debt in developing economies?",
     "chapter": "ch3", "context_facts": [r"70 percent"], "all_chapter": "ch3"},
    {"id": "Q3", "q": "According to the executive summary, what are the main risks to the global outlook?",
     "chapter": "exec", "facts": [r"downside", r"escalation"], "all_chapter": "exec"},
    # --- regions that all talk about growth forecasts ---------------------
    {"id": "R1", "q": "What is the growth outlook for East Asia and Pacific?",
     "chapter": "ch2", "region": "East Asia and Pacific", "facts": [r"4\.2 percent"],
     "forbid": forbid_except(r"(?<![\d.])4\.2 percent")},
    {"id": "R2", "q": "What is the growth outlook for Europe and Central Asia?",
     "chapter": "ch2", "region": "Europe and Central Asia", "facts": [r"2\.1 percent"],
     "forbid": forbid_except(r"(?<![\d.])2\.1 percent")},
    {"id": "R3", "q": "What growth does the report project for Latin America and the Caribbean in 2026?",
     "chapter": "ch2", "region": "Latin America and the Caribbean", "facts": [r"2\.2 percent"],
     "forbid": forbid_except(r"(?<![\d.])2\.2 percent")},
    {"id": "R4", "q": "How is the Middle East and North Africa expected to grow in 2026?",
     "chapter": "ch2", "region": "Middle East, North Africa, Afghanistan, and Pakistan", "facts": [r"1\.6 percent"],
     "forbid": forbid_except(r"1\.6 percent")},
    {"id": "R5", "q": "What is the growth forecast for Sub-Saharan Africa?",
     "chapter": "ch2", "region": "Sub-Saharan Africa", "facts": [r"(?<![\d.])4 percent"],
     # 1.6 percent is excluded: it is also SSA's own per-capita growth figure
     "forbid": [r"6\.3 percent", r"(?<![\d.])4\.2 percent", r"(?<![\d.])2\.1 percent"]},
    # --- routing by country: the report puts Pakistan with the Middle East, not South Asia
    {"id": "R6", "q": "What is the economic outlook for Pakistan?",
     "chapter": "ch2", "region": "Middle East, North Africa, Afghanistan, and Pakistan", "facts": [],
     "forbid": [r"6\.3 percent"]},
    # --- other structure ---------------------------------------------------
    {"id": "C4", "q": "What does Chapter 4 say about fiscal policy in commodity exporters?",
     "chapter": "ch4", "facts": [r"procyclical"], "all_chapter": "ch4"},
    {"id": "G1", "q": "What is the global growth forecast for 2026?",
     "chapter": None, "facts": [r"2\.5 percent"]},
    {"id": "B1", "q": "According to the box on AI, by how much did customer-support agents in the Philippines raise their productivity?",
     "chapter": "ch1", "box": "BOX 1.1", "facts": [r"15 percent"]},
    # --- must refuse -------------------------------------------------------
    {"id": "X1", "q": "What does the report forecast for global GDP growth in 2035?",
     "refuse": True},
]


def matches(hit, case):
    p = hit["parent"]
    if case.get("chapter") and p["chapter_key"] != case["chapter"]:
        return False
    if case.get("region") and p["region"] != case["region"]:
        return False
    if case.get("box") and not (p["box"] or "").startswith(case["box"]):
        return False
    return all(re.search(f, p["text"]) for f in case.get("facts", []))


def retrieval_eval(pipe, no_filters=False):
    rows, rr = [], []
    scored = [c for c in CASES if not c.get("refuse")]
    for c in scored:
        if no_filters:
            plan = {"region": None, "chapter": None, "tables": False}
            hits = retrieve(pipe.index, c["q"], plan, use_reranker=pipe.use_reranker)
        else:
            plan, hits = pipe.retrieve(c["q"])
        rank = next((i for i, h in enumerate(hits, 1) if matches(h, c)), None)
        top = hits[0]["parent"] if hits else None
        scope_ok = True
        if c.get("all_chapter"):
            scope_ok = all(h["parent"]["chapter_key"] == c["all_chapter"] for h in hits)
        ctx_ok = all(any(re.search(f, h["parent"]["text"]) for h in hits) for f in c.get("context_facts", []))
        ok = rank is not None and rank <= 3 and scope_ok and ctx_ok
        rr.append(1.0 / rank if rank else 0.0)
        rows.append((c["id"], ok, rank, top["path"] if top else "-", top["page_start"] if top else "-", plan, scope_ok))
    return rows, rr


def print_retrieval(title, rows, rr):
    print(f"\n=== {title} ===")
    for cid, ok, rank, path, page, plan, scope_ok in rows:
        route = f"region={plan.get('region')} chapter={plan.get('chapter')} [{plan.get('source')}]"
        print(f"[{cid}] {'PASS' if ok else 'FAIL'} rank={rank}  p.{page}  {path[:95]}")
        if not ok:
            print(f"      route: {route}  scope_ok={scope_ok}")
    in_ctx = sum(1 for r in rows if r[1])
    top1 = sum(1 for r in rows if r[1] and r[2] == 1)
    print(f"Right section in the 3 passages sent to the model: {in_ctx}/{len(rows)}   top-1: {top1}/{len(rows)}   MRR: {sum(rr) / len(rr):.3f}")
    return in_ctx == len(rows)


def generation_eval(pipe):
    print("\n=== GENERATION (Gemini) ===")
    passed = 0
    for c in CASES:
        res = pipe.query(c["q"])
        ans = res["answer"]
        if c.get("refuse"):
            ok = REFUSAL.lower() in ans.lower()
            why = "" if ok else "expected refusal"
        else:
            missing = [f for f in c.get("facts", []) if not re.search(f, ans)]
            leaked = [f for f in c.get("forbid", []) if re.search(f, ans)]
            ok = not missing and not leaked
            why = (f"missing {missing} " if missing else "") + (f"leaked {leaked}" if leaked else "")
        passed += ok
        print(f"[{c['id']}] {'PASS' if ok else 'FAIL'} {why}\n      Q: {c['q']}\n      A: {ans[:420].replace(chr(10), ' ')}")
    print(f"Answers correct: {passed}/{len(CASES)}")
    return passed == len(CASES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--no-llm-routing", action="store_true")
    args = ap.parse_args()

    pipe = GEPPipeline(use_llm_routing=not args.no_llm_routing)
    ok = print_retrieval("RETRIEVAL (full pipeline)", *retrieval_eval(pipe))
    if args.ablate:
        print_retrieval("RETRIEVAL (filters off: same index, no region/chapter routing)",
                        *retrieval_eval(pipe, no_filters=True))
    if not args.retrieval_only:
        ok = generation_eval(pipe) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
