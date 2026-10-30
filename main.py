#!/usr/bin/env python3
"""CLI for the Global Economic Prospects (June 2026) RAG pipeline."""

import argparse

from src.pipeline import GEPPipeline


def show(res):
    print(f"\nPLAN: region={res['plan'].get('region')} chapter={res['plan'].get('chapter')} "
          f"tables={res['plan'].get('tables')} (via {res['plan'].get('source')})")
    print("\nRETRIEVED SECTIONS:")
    for i, h in enumerate(res["hits"], 1):
        p = h["parent"]
        print(f"  [{i}] p.{p['page_start']}-{p['page_end']}  {p['path']}  (score {h['score']:.3f})")
    print("\nANSWER:\n" + res["answer"])
    print(f"\n[retrieval {res['retrieval_time']:.2f}s | generation {res['generation_time']:.2f}s]")


def main():
    ap = argparse.ArgumentParser(description="GEP June 2026 RAG")
    ap.add_argument("-q", "--query")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-llm-routing", action="store_true")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    pipe = GEPPipeline(reindex=args.reindex, use_reranker=not args.no_rerank,
                       use_llm_routing=not args.no_llm_routing)
    if args.query:
        res = pipe.query(args.query)
        show(res)
        if args.show_context:
            print("\n" + res["context"])
        return
    print("Ask a question (quit to exit)")
    while True:
        try:
            q = input("\nQuestion > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            show(pipe.query(q))


if __name__ == "__main__":
    main()
