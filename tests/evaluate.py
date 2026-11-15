"""Measure answer recall and abstention on tests/eval_set.py.

Run from the repo root:  python -m tests.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.pipeline import RAGPipeline
from tests import eval_set

DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
HERE = Path(__file__).resolve().parent


def frozen_set(name: str):
    """(answerable (question, fact) pairs, unanswerable questions) from a frozen JSON set."""
    items = json.loads((HERE / name).read_text())
    return (
        [(it["q"], it["expected"]) for it in items if it["answerable"]],
        [it["q"] for it in items if not it["answerable"]],
    )


def build_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline()
    pipeline.index_directory(DOCS)
    return pipeline


def recall(pipeline: RAGPipeline, items) -> list[str]:
    """Questions whose answer does not contain the expected fact (abstaining counts as a miss)."""
    misses = []
    for question, expected in items:
        r = pipeline.query(question)
        if r.abstained or expected.lower() not in r.answer.lower():
            misses.append(question)
    return misses


def answered(pipeline: RAGPipeline, questions) -> list[str]:
    """Out-of-corpus questions that were answered instead of abstained on."""
    return [q for q in questions if not pipeline.query(q).abstained]


def main() -> None:
    p = build_pipeline()
    for name in ("KNOWN", "HELD_OUT", "FRESH", "FRESH_2"):
        items = getattr(eval_set, name)
        misses = recall(p, items)
        print(f"{name}: {len(items) - len(misses)}/{len(items)} answers contain the expected fact")
        for q in misses:
            print(f"   miss: {q}")
    for name in ("OUT_OF_CORPUS", "FRESH_OUT_OF_CORPUS", "NEAR_TOPIC"):
        qs = getattr(eval_set, name)
        wrong = answered(p, qs)
        print(f"{name}: {len(qs) - len(wrong)}/{len(qs)} abstained")
        for q in wrong:
            print(f"   answered: {q}")
    for name in ("answerability_set.json", "answerability_test2.json"):
        ans, unans = frozen_set(name)
        refused = [q for q, _ in ans if p.query(q).abstained]
        print(f"{name}: {len(unans) - len(answered(p, unans))}/{len(unans)} unanswerable abstained, "
              f"{len(refused)}/{len(ans)} answerable wrongly refused, "
              f"{len(ans) - len(recall(p, ans))}/{len(ans)} answers contain the expected fact")


if __name__ == "__main__":
    main()
