# Experiments

## Intake urgency: "any symptom means Priority or higher" rule

- **Change:** added a rule to the intake prompt saying that if the patient
  describes any symptom, urgency must be Priority or higher.
- **Baseline:** urgency accuracy 0.75 (15/20) on the 20-record intake set.
- **After change:** urgency accuracy 0.30 (6/20). Other fields unchanged.
- **Decision:** reverted. The answer key separates Routine from Priority
  by how acute the problem sounds, so "any symptom" over-escalates
  (for example chronic sinus infections are Routine in the key).
- **Evidence:** `evals/results/intake_eval.json` (baseline) and
  `evals/results/intake_eval_experiment_symptom_rule.json` (experiment).
- **Known issue:** record CAREFLOW-00007 has `specialty` listed as missing
  even though the message states ENT. This looks like an answer-key error,
  so we did not tune the prompt to it.