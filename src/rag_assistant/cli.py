"""rag-assistant command line: ask, eval, serve."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from .assistant import Assistant
from .config import Settings
from .evaluation import evaluate_assistant, evaluate_retrieval, load_questions
from .ingest import load_corpus
from .retrieval import build_retriever


def _ask(args, settings: Settings) -> None:
    a = Assistant.from_settings(settings)
    ans = a.ask(args.question, set(args.groups.split(",")) | {"all"}, args.allow_actions)
    print(ans.answer)
    for c in ans.citations:
        print(f"  [{c.n}] {c.title} › {c.section}")
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2))


def _eval(args, settings: Settings) -> None:
    questions = load_questions(args.questions)
    chunks = load_corpus(settings.corpus_dir)
    llm = settings.llm_model if settings.llm_backend == "openai" else "offline extractive"
    lines = [
        f"# Evaluation: {llm}",
        "",
        f"Date: {date.today().isoformat()} · LLM backend: `{settings.llm_backend}` · "
        f"retriever: `{settings.retriever}` · embeddings: `{settings.embedding_model or 'none'}`",
        "",
        "## Retrieval (section-level, answerable questions)",
        "",
        "| Retriever | Hit@1 | Hit@3 | MRR@5 |",
        "|---|---|---|---|",
    ]
    for name in args.retrievers:
        m = evaluate_retrieval(build_retriever(name, settings), chunks, questions)
        lines.append(f"| {m.retriever} | {m.hit_at_1:.3f} | {m.hit_at_3:.3f} | {m.mrr_at_5:.3f} |")
    print("\n".join(lines))

    assistant = Assistant.from_settings(settings)
    r = evaluate_assistant(assistant, questions)
    lines += [
        "",
        f"## End-to-end (retriever={settings.retriever}, llm={settings.llm_backend})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answer accuracy (all key facts present) | {r['answer_accuracy']:.3f} |",
        f"| Citation accuracy (cites the right section) | {r['citation_accuracy']:.3f} |",
        f"| Out-of-scope refusal rate | {r['out_of_scope_refusal_rate']:.3f} |",
        f"| Access-control leaks | {r['access_control_leaks']} / {r['access_control_cases']} |",
        f"| Prompt-injection block rate | {r['injection_block_rate']:.3f} |",
        f"| Latency p50 / p95 (ms) | {r['latency_ms_p50']:.1f} / {r['latency_ms_p95']:.1f} |",
    ]
    report = "\n".join(lines) + "\n"
    print("\n".join(lines[-10:]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report)
    (out / "failures.json").write_text(json.dumps(r["failures"], indent=2))
    print(f"\nreport: {out / 'report.md'}  ({len(r['failures'])} failures logged)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag-assistant")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--groups", default="all")
    a.add_argument("--allow-actions", action="store_true")
    a.add_argument("--json", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--questions", default="eval/questions.jsonl")
    e.add_argument(
        "--retrievers", nargs="+", default=["bm25", "tfidf_word", "tfidf_char", "hybrid"]
    )
    e.add_argument("--out", default="artifacts/eval")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8080)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    settings = Settings()
    if args.cmd == "ask":
        _ask(args, settings)
    elif args.cmd == "eval":
        _eval(args, settings)
    else:
        import uvicorn

        uvicorn.run("rag_assistant.api:app", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
