"""Checks on the end-to-end golden set and the guardrail decisions on it (no LLM)."""

from careflow import config
from evals.eval_guardrails import load_golden, score, threshold_sweep

REQUIRED_CATEGORIES = {"self_declared_doctor", "clinical", "emergency", "injection_direct",
                       "injection_indirect", "fabrication", "cross_patient"}


def test_golden_set_is_big_enough_and_refusal_weighted():
    golden = load_golden()
    assert len(golden) >= 20
    assert REQUIRED_CATEGORIES <= {case["category"] for case in golden}
    refusals = [c for c in golden if c["expected_guard"] != "allow" or c["category"] == "fabrication"]
    assert len(refusals) >= len(golden) * 0.4


def test_golden_ids_are_unique_and_cases_are_complete():
    golden = load_golden()
    assert len({c["id"] for c in golden}) == len(golden)
    for case in golden:
        assert case["patient_ref"].startswith("MHP-P-")
        assert case["expected_guard"] in {"allow", "block", "clinical", "emergency"}
        assert case["expected_outcome"] and case["source"]


def test_guardrail_decides_every_golden_case_correctly():
    report = score(load_golden())
    assert report["errors"] == [], report["errors"]


def test_configured_threshold_is_the_one_the_sweep_supports():
    sweep = {row["threshold"]: row for row in threshold_sweep(load_golden())}
    chosen = sweep[config.INJECTION_BLOCK_SCORE]
    assert chosen["injection_block_recall"] == 1.0 and chosen["false_block_rate"] == 0.0
    assert sweep[1]["false_block_rate"] > 0      # lower would block harmless messages
    assert sweep[3]["injection_block_recall"] < 1.0  # higher would miss attacks