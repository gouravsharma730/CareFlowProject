"""Score the input guardrail on the end-to-end golden set (no LLM calls).

For every golden message the guardrail decides allow / block / clinical /
emergency, and that decision is compared with the expected one.

Reported:
  accuracy and a confusion table (expected vs decided)
  per decision: precision and recall
  refusal precision / recall (domain metric): a "refusal" is any decision
    that stops the request from being answered normally (block, clinical,
    emergency). Recall = unsafe requests caught; precision = refusals that
    were justified.
  injection-threshold sweep: block recall and false-block rate for
    INJECTION_BLOCK_SCORE = 1..4, which is how the threshold was chosen.

Run:
    python -m evals.eval_guardrails

Results are saved to evals/results/guardrail_eval.json.
"""

import json
from collections import Counter
from datetime import datetime, timezone

from careflow import config, guardrails

GOLDEN_PATH = config.PROJECT_ROOT / "evals" / "golden" / "e2e_golden.jsonl"
RESULTS_PATH = config.PROJECT_ROOT / "evals" / "results" / "guardrail_eval.json"
DECISIONS = ("allow", "block", "clinical", "emergency")
REFUSALS = {"block", "clinical", "emergency"}


def load_golden(path=GOLDEN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def score(golden: list[dict]) -> dict:
    rows = []
    for case in golden:
        check = guardrails.check_input(case["message"])
        rows.append({"id": case["id"], "category": case["category"],
                     "expected": case["expected_guard"], "decided": check.action,
                     "correct": check.action == case["expected_guard"],
                     "flags": check.flags, "injection_score": check.injection_score})
    confusion = Counter((r["expected"], r["decided"]) for r in rows)
    per_decision = {}
    for d in DECISIONS:
        tp = confusion[(d, d)]
        decided = sum(v for (e, x), v in confusion.items() if x == d)
        expected = sum(v for (e, x), v in confusion.items() if e == d)
        per_decision[d] = {"precision": _ratio(tp, decided), "recall": _ratio(tp, expected),
                           "support": expected}
    should_refuse = [r for r in rows if r["expected"] in REFUSALS]
    did_refuse = [r for r in rows if r["decided"] in REFUSALS]
    return {
        "accuracy": _ratio(sum(r["correct"] for r in rows), len(rows)),
        "refusal_recall": _ratio(sum(r["decided"] in REFUSALS for r in should_refuse), len(should_refuse)),
        "refusal_precision": _ratio(sum(r["expected"] in REFUSALS for r in did_refuse), len(did_refuse)),
        "per_decision": per_decision,
        "confusion": {f"{e} -> {x}": v for (e, x), v in sorted(confusion.items())},
        "errors": [r for r in rows if not r["correct"]],
        "rows": rows,
    }


def threshold_sweep(golden: list[dict], thresholds=(1, 2, 3, 4)) -> list[dict]:
    """Block recall on injection cases and false blocks on everything else."""
    injections = [c for c in golden if c["expected_guard"] == "block"]
    others = [c for c in golden if c["expected_guard"] != "block"]
    sweep = []
    for threshold in thresholds:
        blocked = lambda c: guardrails.injection_scan(c["message"])[0] >= threshold
        sweep.append({
            "threshold": threshold,
            "injection_block_recall": _ratio(sum(blocked(c) for c in injections), len(injections)),
            "false_block_rate": _ratio(sum(blocked(c) for c in others), len(others)),
            "false_blocks": [c["id"] for c in others if blocked(c)],
        })
    return sweep


def main() -> dict:
    golden = load_golden()
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden_set": str(GOLDEN_PATH.relative_to(config.PROJECT_ROOT)),
        "cases": len(golden),
        "injection_block_score": config.INJECTION_BLOCK_SCORE,
        **score(golden),
        "threshold_sweep": threshold_sweep(golden),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Cases: {report['cases']}   accuracy: {report['accuracy']}   "
          f"refusal recall: {report['refusal_recall']}   refusal precision: {report['refusal_precision']}")
    print("\nPer decision:")
    for name, m in report["per_decision"].items():
        print(f"  {name:10} precision {m['precision']}  recall {m['recall']}  (n={m['support']})")
    print("\nConfusion (expected -> decided):")
    for key, value in report["confusion"].items():
        print(f"  {key:26} {value}")
    print("\nInjection threshold sweep:")
    for row in report["threshold_sweep"]:
        print(f"  score >= {row['threshold']}: block recall {row['injection_block_recall']}, "
              f"false-block rate {row['false_block_rate']} {row['false_blocks']}")
    if report["errors"]:
        print("\nMistakes:")
        for r in report["errors"]:
            print(f"  {r['id']} ({r['category']}): expected {r['expected']}, got {r['decided']} {r['flags']}")
    print(f"\nSaved to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    main()