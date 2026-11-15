# RAG Compliance Intelligence

**Answers questions over AML, GDPR and SOX sample documents with cited sentences: 14 of 14 on a known-answer set. On 117 new test questions frozen before scoring, it refused 58 of 60 about nearby rules the documents never cover (51 of 60 before this version) and wrongly refused 8 of 57 answerable ones (10 of 57 before). Fast answers took about 2 s each on the free CPU; optional small-model answers about 17 s.**

Ask a question about the bundled sample documents (an AML policy, a GDPR compliance document, SOX
internal controls, and public summaries of HIPAA, CCPA/CPRA and PCI DSS) and get an answer with cited
passages. By default the answer is written by the `Qwen/Qwen2.5-1.5B-Instruct` language model, grounded
in the retrieved passages; it falls back to fast extractive answering if the model cannot load, so the
Space keeps working on the free CPU. The in-app "How it works & measured evaluation" panel surfaces the
retrieval stack, the 0.92 abstention threshold rationale, and the measured recall/abstention numbers.

## Update (2026-11-06): corpus expanded, abstain threshold retuned

The corpus grew from 3 to 6 documents (HIPAA, CCPA/CPRA and PCI DSS summaries added, sourced in each file
header) and the abstain threshold was retuned from 0.785 to 0.92 on a new adversarial near-topic set
(`tests/adversarial_eval.json`). Previously two near-topic questions the documents do not cover were
answered instead of refused — "What will the EU Anti-Money Laundering Authority supervise?" and "How does
Singapore's Personal Data Protection Act handle breach notification?"; both now refuse. On the adversarial
set: near-topic out-of-corpus refused 19/20, answerable (original docs) wrongly refused 0/15, answerable
(added docs) answered 13/14, known in-corpus accuracy 14/14. Tradeoff: raising the threshold lifts wrong
refusals on the repo's 212 answerable questions from 13/212 (6%) to 24/212 (11%). Reproduce with
`python -m tests.evaluate_adversarial`. The tables below describe the original three-document baseline at
threshold 0.785.

## How it works

1. **Retrieval:** the documents are split into passages, embedded with `sentence-transformers/all-MiniLM-L6-v2`
   and searched with a FAISS `IndexFlatIP` index. Each passage's cosine similarity is blended 50/50 with its
   BM25 keyword score (max-normalised), and the same blend ranks the individual sentences of the top passages.
2. **Abstain:** the app says the answer is not in the loaded documents, and gives none, when the question names
   something the documents never use (a capitalised word or acronym such as HIPAA or FINRA), or when its
   answerability score is below 0.785. The score is a logistic combination of the best sentence cosine similarity
   and the best scores the cross-encoder `cross-encoder/ms-marco-MiniLM-L-12-v2` gives the question paired with each
   top sentence and with each of the five top passages. A cross-encoder reads the question and the text together,
   so it can tell a sentence that answers the question from one that only shares its words.
3. **Answer, language-model mode (default):** `Qwen/Qwen2.5-1.5B-Instruct` writes a short answer from the five best
   sentences only, told to say so when they lack the answer. Numbers in its answer that do not appear in those
   sentences are flagged (`rag/llm_generator.py:_flag_unverified`, covered by `test_number_guard_flags_fabricated_numbers`).
   The model loads in the background at startup; if the 3 GB model fails to load, is not ready, or does not finish
   within 60 seconds, the fast extractive answer is shown with a note saying so, so the Space still works on CPU.
4. **Answer, fast extractive mode (uncheck the box):** the three best sentences are quoted with their heading and
   source. No language model is involved — this is also the automatic fallback above.
5. **Citations:** each quoted sentence comes with a `[document §section]` marker naming the section that
   sentence sits in. The app surfaces these cited passages and the compliance regime(s) the answer is grounded in
   directly beneath every answer, and abstention never cites a passage.

## What was tested

`tests/eval_set.py` holds the questions; `python -m tests.evaluate` prints the numbers below and
`python -m pytest tests` checks them. An answer counts as correct when it contains the expected fact from the
documents (for example "one month" for the access-request deadline).

| Question set | Dense retrieval only | Hybrid retrieval and similarity check |
|---|---|---|
| 14 first questions | 12/14 | 14/14 |
| 44 held-out questions | 28/44 | 41/44 |
| 22 fresh questions, written after the settings were fixed | 17/22 | 22/22 |
| 40 questions the documents cannot answer: abstained | 0/40 | 39/40 |
| 20 fresh unanswerable questions: abstained | 0/20 | 19/20 |

The last column used the earlier abstain rule (sentence similarity of at least 0.50, described here; replaced by
the answerability score below). The 0.50 threshold maximises Youden's J (true-answer rate minus false-answer rate) on the 58 answerable and 40
unanswerable questions: 1 of 58 answerable questions is wrongly refused and 1 of 40 unanswerable ones is answered.
Re-choosing the threshold on random halves and testing on the other half (50 repeats) gave 4.5% wrong refusals and
4.1% wrong answers. The 50/50 blend was not tuned: every weight from 0.2 to 0.8 gave 14/14 and 41-42/44 before
abstention, dense-only gave 11/14 and 38/44.

The name check was added afterwards. It was built on the sets above, where no answerable question names
something the documents lack, then tested once on two new sets written before it was built: 35 questions about
nearby rules and standards the documents never mention (PSD2, FINRA, the Corporate Transparency Act, ...) and 22
new answerable questions.

| Question set | Similarity check only | Plus name check |
|---|---|---|
| 35 nearby-topic questions: abstained | 24/35 | 31/35 |
| 40 + 20 unanswerable questions above: abstained | 58/60 | 59/60 |
| 22 new answerable questions: correct | 20/22 | 20/22 |
| All 102 answerable questions: wrongly refused | 3/102 | 3/102 |

The name check refused none of the 102 answerable questions; all three wrong refusals come from the 0.50
similarity threshold, and two of them are in the new set, so the earlier 1-in-58 estimate was optimistic.

### Answerability score

The similarity rule still answered nearby-rule questions written only in words the documents contain ("How does
the SEC regulate investment advisers?", the EU's sixth anti-money laundering directive, the Corporate Transparency
Act, EU Regulation 833/2014, the Bank Holding Company Act). The answerability score replaced it. Its weights and its
cut were fitted on every earlier question: the sets above plus `tests/answerability_set.json` (76 nearby-rule and
53 answerable questions written for this check). The cut is the one that refuses the most of those questions while
answering all 14 known questions and wrongly refusing no more than the 0.50 rule did. On that fitting data, which is
in-sample, it refused 171 of 171 unanswerable questions and wrongly refused 5 of 155 answerable ones (11 of 155
before); on the 102 answerable questions in the first tables it wrongly refused 2 (3 before).

`tests/answerability_test2.json` (60 nearby-rule and 57 answerable questions, sha256 `1fd07194d634…`) was written
and frozen before the weights were chosen, then scored once. The live Space gave the same counts.

| 117 frozen test questions | Before (similarity 0.50 + name check) | After (answerability score + name check) |
|---|---|---|
| 60 about nearby rules: refused | 51/60 = 85% (95% CI 74-92%) | 58/60 = 97% (89-99%) |
| 57 answerable: wrongly refused | 10/57 = 18% (10-29%) | 8/57 = 14% (7-25%) |
| 57 answerable: answer contains the expected fact | 43/57 | 44/57 |

Confidence intervals are Wilson intervals. Paired on the same questions (bootstrap 95% CI), refusals of nearby-rule
questions rose by 11.7 points (+5.0 to +20.0; 7 newly refused, none lost) and wrong refusals changed by -3.5 points
(-12.3 to +5.3), so the drop in wrong refusals is within noise.

An earlier attempt (`cross-encoder/ms-marco-MiniLM-L-6-v2` on sentences only) was fitted on part of
`answerability_set.json` and scored once on the rest. It refused 46 of 46 nearby-rule questions (38 before) but
wrongly refused 3 of 32 answerable ones (2 before) and one of the 14 known questions, so it was not shipped. That set
then became fitting data and the new test set was written.

## Limits

- Fast mode quotes sentences; it does not reason over them and can quote related but unhelpful text.
- Some questions about nearby rules are still answered (in the test set: "What will the EU Anti-Money Laundering
  Authority supervise?", "How does Singapore's Personal Data Protection Act handle breach notification?").
- About 1 in 7 answerable questions in the test set was wrongly refused, mostly short ones (refused: "Who issues
  adequacy decisions?", "Is trade union membership sensitive data?", "Which wire transfers are red flags?").
- The answerability score makes fast mode slower: on this Space, the 117 test questions took a median of 1.8 s each on the
  server (95th percentile 2.3 s, slowest 4.0 s).
- The name check reads capital letters. In a question typed in Title Case it only looks at acronyms, so a name
  like "Bank Holding Company Act" written that way is not caught.
- The test questions were written by the author from the same three documents; they are not an independent
  benchmark.
- Language-model mode can still be wrong or misread a passage. Check the cited sources.
- Language-model mode is slow on the free CPU: three questions measured on this Space took 14.5, 16.9 and 17.3
  seconds each (the model was already loaded), and one more after the answerability change took 16.7 seconds. After a restart the model downloads (3.1 GB) and loads in the
  background; that took 14 seconds when measured. Over 60 seconds, the passage-based answer is shown.
- "Retrieval similarity" measures how close the passages are to the question, not whether the answer is right.
- The documents are samples bundled with the demo. This is not legal advice or current regulation text.

Source: the Files tab of this Space
