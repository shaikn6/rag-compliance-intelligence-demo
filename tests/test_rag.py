"""Accuracy and abstention checks on the bundled documents.

Run from the repo root:  python -m pytest tests
"""

import re

import pytest

from rag.document_loader import DocumentLoader
from rag.llm_generator import LLMGenerator, _flag_unverified
from rag.retriever import split_sentences
from tests import eval_set
from tests.evaluate import DOCS, answered, build_pipeline, frozen_set, recall


# Regulations added to the corpus on 2026-11-06 (HIPAA, CCPA/CPRA, PCI DSS).
# Questions about them that older frozen sets label "out of corpus" are now
# answerable, so they are excluded from the abstention assertions below.
NOW_COVERED = (
    "hipaa", "health insurance portability",
    "ccpa", "california consumer privacy", "california privacy rights", "cpra",
    "pci", "payment card industry", "card industry data security",
)


def _still_out_of_corpus(questions):
    return [q for q in questions if not any(t in q.lower() for t in NOW_COVERED)]


@pytest.fixture(scope="module")
def pipeline():
    return build_pipeline()


def test_known_questions_all_answered(pipeline):
    assert recall(pipeline, eval_set.KNOWN) == []


def test_held_out_recall(pipeline):
    # 42/44 when measured; allow one regression before failing.
    assert len(recall(pipeline, eval_set.HELD_OUT)) <= 3


def test_out_of_corpus_abstains(pipeline):
    for q in ("What is the capital of France?", "What does MiFID II require for trade reporting?"):
        r = pipeline.query(q)
        assert r.abstained and r.citations == [] and "Not found" in r.answer
    # Excludes questions now covered by the added documents; 0/35 answered when measured at threshold 0.92.
    assert len(answered(pipeline, _still_out_of_corpus(eval_set.OUT_OF_CORPUS))) <= 2


def test_answerable_questions_not_abstained(pipeline):
    items = eval_set.KNOWN + eval_set.HELD_OUT
    assert sum(pipeline.query(q).abstained for q, _ in items) <= 2


def test_split_sentences_keeps_headings_as_labels():
    text = (
        "3.1 Right of Access (Article 15)\n"
        "Data subjects may get a copy. Requests must be fulfilled\n"
        "within one month.\n"
        "SECTION 4: DATA BREACH NOTIFICATION\n"
        "4.1 Internal Detection\n"
        "components: 1. Control Environment: tone."
    )
    s3, s4 = "SECTION 3: DATA SUBJECT RIGHTS", "SECTION 4: DATA BREACH NOTIFICATION"
    assert split_sentences(text, s3) == [
        (s3, "3.1 Right of Access (Article 15)", "Data subjects may get a copy."),
        (s3, "3.1 Right of Access (Article 15)", "Requests must be fulfilled within one month."),
        (s4, "4.1 Internal Detection", "components: 1. Control Environment: tone."),
    ]


def test_every_sentence_is_labelled_with_its_own_section():
    """The section given for a sentence is the last SECTION heading above it in the document."""
    docs = {p.stem: p.read_text() for p in DOCS.glob("*.txt")}
    for chunk in DocumentLoader().load_directory(DOCS):
        doc = docs[chunk.source_doc]
        for section, _, sentence in split_sentences(chunk.text, chunk.start_section):
            pattern = r"\s+".join(map(re.escape, sentence.split()))
            pos = re.compile(pattern).search(doc, chunk.char_start).start()
            above = re.findall(r"^SECTION \d+:.*$", doc[:pos], re.M)
            expected = above[-1].strip() if above else "General"  # title lines before SECTION 1
            assert section == expected, (chunk.chunk_id, sentence[:60])


def test_citation_names_the_cited_sentences_section(pipeline):
    r = pipeline.query("How quickly must access be revoked for terminated employees under SOX ITGC?")
    assert "24 hours" in r.answer
    assert r.citations[0].marker == "[sox_controls §SECTION 4: IT GENERAL CONTROLS (ITGCs)]"
    assert "Terminated employees" in r.citations[0].excerpt


def test_unknown_names_abstain(pipeline):
    # FINRA is not in the corpus; the name check must catch it (HIPAA now is in the corpus).
    r = pipeline.query("What are the FINRA rules for broker-dealers?")
    assert r.abstained and "FINRA" in r.answer
    # Names the documents do use do not trigger it.
    assert pipeline._retriever.unknown_terms("Which SOX section covers CEO and CFO certification?") == []
    # In Title Case only acronyms count as names.
    assert not pipeline.query("How Quickly Must Access Be Revoked For Terminated Employees?").abstained


def test_nearby_rules_in_document_words_abstain(pipeline):
    """Questions about rules the documents never cover, worded with words they do use."""
    for q in (
        "How does the SEC regulate investment advisers?",
        "What does the EU's sixth anti-money laundering directive add to money laundering offences?",
        "What does the Corporate Transparency Act require for beneficial ownership reports to FinCEN?",
        "What does EU Regulation 833/2014 require for sanctions screening?",
        "What activities does the Bank Holding Company Act restrict?",
    ):
        r = pipeline.query(q)
        assert r.abstained and r.citations == [], q
    r = pipeline.query("Which article gives the right to object?")
    assert not r.abstained and "Article 21" in r.answer


def test_frozen_answerability_sets(pipeline):
    # Re-measured on the 6-document corpus at threshold 0.92 (HIPAA/CCPA/PCI
    # questions in the unanswerable partition are now in-corpus and excluded):
    #   set1  still-out-of-corpus answered 0/73, answerable wrongly refused 6/53;
    #   test2 still-out-of-corpus answered 0/60, answerable wrongly refused 12/57.
    # The higher wrong-refusal bound is the cost of raising the threshold from 0.785
    # to 0.92 to refuse near-topic offenders (EU AMLA, Singapore PDPA).
    for name, max_answered, max_refused in (
        ("answerability_set.json", 1, 8),
        ("answerability_test2.json", 1, 14),
    ):
        ans, unans = frozen_set(name)
        assert len(answered(pipeline, _still_out_of_corpus(unans))) <= max_answered, name
        assert sum(pipeline.query(q).abstained for q, _ in ans) <= max_refused, name


def test_llm_mode_falls_back_when_model_unavailable(pipeline):
    llm = LLMGenerator(model_id="not-a-real/model-id")
    llm.start_loading()
    results = pipeline._retriever.retrieve("What is the CTR threshold?")
    a = llm.generate("What is the CTR threshold?", results)
    assert "10,000" in a.answer and "passage-based answer" in a.answer


def test_number_guard_flags_fabricated_numbers():
    """A number the model writes that is not in the cited passages is flagged."""
    context = "[1] (aml_policy) Transactions over $10,000 require a Currency Transaction Report."
    # Faithful answer: the only number (10,000) is in the context, nothing flagged.
    assert _flag_unverified("The CTR threshold is $10,000.", context, 1) == "The CTR threshold is $10,000."
    # Fabricated answer: 25,000 is not in the context, so it is flagged.
    flagged = _flag_unverified("The CTR threshold is $25,000.", context, 1)
    assert "⚠️" in flagged and "25000" in flagged
    # Passage index markers [1]..[n] are not treated as fabricated numbers.
    assert _flag_unverified("See passage [1].", context, 1) == "See passage [1]."
