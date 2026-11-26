"""Score the intake extraction against the ground truth in records.jsonl.

Usage (from the project root):
    python -m evals.eval_intake --limit 5     # quick try
    python -m evals.eval_intake               # all records
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from careflow import config
from careflow.intake import extract_patient_request

logging.basicConfig(level=logging.WARNING)

RECORDS_PATH = config.DATA_DIR / "careflow" / "intake" / "records.jsonl"
RESULTS_PATH = config.PROJECT_ROOT / "evals" / "results" / "intake_eval.json"

CHECKED_FIELDS = ["urgency", "specialty", "seeks_clinical_advice", "missing_fields"]


def load_records(limit):
    records = []
    with open(RECORDS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records[:limit] if limit else records


def same(field, predicted, expected):
    """Compare one field. missing_fields is compared as a set."""
    if field == "missing_fields":
        return set(predicted or []) == set(expected or [])
    return predicted == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = load_records(args.limit)
    print(f"Scoring {len(records)} records with model {config.LLM_MODEL}")

    field_correct = {field: 0 for field in CHECKED_FIELDS}
    by_channel = defaultdict(lambda: {"total": 0, "valid": 0, "all_correct": 0})
    failures = []
    valid_count = 0

    for number, record in enumerate(records, start=1):
        channel = record["channel"]
        by_channel[channel]["total"] += 1
        try:
            result = extract_patient_request(record)
        except Exception as error:  # a failed record is counted, not fatal
            failures.append({"record_id": record["record_id"], "error": str(error)[:200]})
            print(f"{number}/{len(records)} {record['record_id']}: FAILED")
            continue

        valid_count += 1
        by_channel[channel]["valid"] += 1
        truth = record["ground_truth"]
        correct_here = 0
        mismatches = {}
        for field in CHECKED_FIELDS:
            predicted = getattr(result, field)
            if same(field, predicted, truth.get(field)):
                field_correct[field] += 1
                correct_here += 1
            else:
                mismatches[field] = {"predicted": predicted, "expected": truth.get(field)}
        if correct_here == len(CHECKED_FIELDS):
            by_channel[channel]["all_correct"] += 1
        elif mismatches:
            failures.append({"record_id": record["record_id"], "mismatches": mismatches})
        print(f"{number}/{len(records)} {record['record_id']}: {correct_here}/{len(CHECKED_FIELDS)} fields correct")
        time.sleep(0.5)  # be gentle with the API rate limit

    total = len(records)
    summary = {
        "model": config.LLM_MODEL,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "records_total": total,
        "records_valid": valid_count,
        "validation_success_rate": round(valid_count / total, 3) if total else None,
        "field_accuracy": {
            field: round(field_correct[field] / valid_count, 3) if valid_count else None
            for field in CHECKED_FIELDS
        },
        "by_channel": dict(by_channel),
        "problem_records": failures,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nValid records:", f"{valid_count}/{total}")
    print("Field accuracy:", summary["field_accuracy"])
    print("Saved to", RESULTS_PATH)


if __name__ == "__main__":
    main()