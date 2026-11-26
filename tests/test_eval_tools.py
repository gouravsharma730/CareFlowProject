"""Tests for the scoring rule used by the tool-selection evaluation."""

from evals.eval_tools import CASES, tool_choice_is_correct


def test_expected_tool_must_be_called():
    assert tool_choice_is_correct("calculate_copay", ["calculate_copay"])
    assert not tool_choice_is_correct("calculate_copay", ["find_appointment_slots"])


def test_no_tool_expected_means_no_tool_called():
    assert tool_choice_is_correct(None, [])
    assert not tool_choice_is_correct(None, ["flag_for_human"])


def test_cases_cover_every_tool():
    expected = {case["expected_tool"] for case in CASES}
    assert {
        "check_insurance_eligibility",
        "find_appointment_slots",
        "calculate_copay",
        "check_preauthorization",
        "flag_for_human",
    } <= expected