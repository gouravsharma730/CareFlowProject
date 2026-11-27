"""Tests for the answer-pipeline scorers, with fake answers and a fake judge."""

import json

from careflow.rag import RagResult
from evals.eval_rag import citation_hit, judge, numbers_match, refusal_correct, run

ANSWERABLE = {
    "id": "x1", "type": "exact_id", "question": "Telehealth co-pay on Bronze?",
    "relevant_chunks": ["copay-schedule#003"], "expected_answer": "$18.00.",
}
UNANSWERABLE = {
    "id": "x2", "type": "unanswerable", "question": "Acupuncture co-pay?",
    "relevant_chunks": [], "expected_answer": "The documents do not say.",
}
CHUNKS = {"copay-schedule#003": {"chunk_id": "copay-schedule#003", "text": "Bronze $18.00"}}


def result(answer, citations, answerable=True):
    return RagResult(
        question="q", answer=answer, answerable=answerable, citations=citations,
        sources=[], retrieved_ids=citations, attempts=1,
    )


def test_deterministic_scorers_on_a_correct_answer():
    good = result("It is $18.00.", ["copay-schedule#003"])
    assert citation_hit(ANSWERABLE, good) is True
    assert refusal_correct(ANSWERABLE, good) is True
    assert numbers_match(ANSWERABLE, good) is True


def test_deterministic_scorers_catch_wrong_number_and_citation():
    bad = result("It is $10.00.", ["copay-schedule#002"])
    assert citation_hit(ANSWERABLE, bad) is False
    assert numbers_match(ANSWERABLE, bad) is False


def test_refusal_scoring():
    refused = result("I don't know.", [], answerable=False)
    assert refusal_correct(UNANSWERABLE, refused) is True
    assert refusal_correct(ANSWERABLE, refused) is False
    assert citation_hit(UNANSWERABLE, refused) is None


def test_judge_retries_once_then_gives_up():
    calls = []

    def broken(messages, json_mode=False):
        calls.append(1)
        return "not json"

    verdict = judge(ANSWERABLE, result("$18.00", []), CHUNKS, broken)
    assert verdict["correct"] is None and len(calls) == 2


def test_run_reports_agreement_between_code_and_judge():
    answers = {
        ANSWERABLE["question"]: result("It is $18.00.", ["copay-schedule#003"]),
        UNANSWERABLE["question"]: result("I don't know.", [], answerable=False),
    }

    def fake_judge(messages, json_mode=False):
        return json.dumps({"grounded": True, "correct": True, "reason": "ok"})

    report = run(
        golden=[ANSWERABLE, UNANSWERABLE],
        answer=lambda q: answers[q],
        llm=fake_judge,
        chunks_by_id=CHUNKS,
    )
    summary = report["summary"]
    assert summary["deterministic_correct_rate"] == 1.0
    assert summary["judge_vs_deterministic_agreement"] == 1.0
    assert summary["disagreements"] == []