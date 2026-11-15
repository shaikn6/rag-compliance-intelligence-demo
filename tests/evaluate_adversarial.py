"""Measure abstention on the adversarial near-topic set (tests/adversarial_eval.json).

These questions sound in-domain but are out of the corpus (EU AMLA, Singapore
PDPA, Volcker Rule, ...), plus genuine answerable questions from the original and
the newly added documents. Reports, at the live ANSWERABLE_THRESHOLD:

  * near-topic out-of-corpus questions correctly refused,
  * answerable questions wrongly refused (over-refusal),
  * accuracy on the 14 known in-corpus questions,

and confirms that the two named offenders (EU AMLA, Singapore PDPA) now refuse.

Run from the repo root:  python -m tests.evaluate_adversarial
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.pipeline import RAGPipeline
from tests import eval_set

DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
ADV = Path(__file__).resolve().parent / "adversarial_eval.json"

# Exact phrasings the Space README flagged as still-answered offenders.
NAMED_OFFENDERS = [
    "What will the EU Anti-Money Laundering Authority supervise?",
    "How does Singapore's Personal Data Protection Act handle breach notification?",
]


def main() -> None:
    p = RAGPipeline()
    n = p.index_directory(DOCS)
    items = json.loads(ADV.read_text())
    thr = p.ANSWERABLE_THRESHOLD
    print(f"{n} chunks indexed · abstain threshold {thr}\n")

    def group(cat):
        return [it for it in items if it["category"] == cat]

    ooc = group("near_topic_ooc")
    refused = sum(p.query(it["q"]).abstained for it in ooc)
    print(f"near-topic out-of-corpus correctly refused: {refused}/{len(ooc)}")
    for it in ooc:
        if not p.query(it["q"]).abstained:
            print(f"   still answered: {it['topic']} — {it['q']}")

    for cat, label in (("answerable_core", "original docs"), ("answerable_new", "added docs")):
        g = group(cat)
        wrong = [it for it in g if p.query(it["q"]).abstained]
        correct = sum(
            (not p.query(it["q"]).abstained)
            and it["expected"].lower() in p.query(it["q"]).answer.lower()
            for it in g
        )
        print(f"answerable ({label}): wrongly refused {len(wrong)}/{len(g)}, "
              f"answered correctly {correct}/{len(g)}")

    known_ok = sum(
        (not p.query(q).abstained) and e.lower() in p.query(q).answer.lower()
        for q, e in eval_set.KNOWN
    )
    print(f"known in-corpus accuracy: {known_ok}/{len(eval_set.KNOWN)}")

    print("\nnamed offenders (should refuse):")
    for q in NAMED_OFFENDERS:
        print(f"   refused={p.query(q).abstained}  {q}")


if __name__ == "__main__":
    main()
