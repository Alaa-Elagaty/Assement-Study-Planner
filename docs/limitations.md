# Limitations

## Correlation, not cause (required)
The neural network learned from **observational** data: students who clicked more also tended to score
higher. That does **not** prove that extra study *causes* higher marks. Weaker or stronger students,
motivation, prior knowledge and course difficulty all influence both clicks and marks. The planner's
statement "N sessions raise the predicted mark to X" is a **what-if estimate under our assumption**,
not a promise. Use it as decision support for students and advisers only.

## How study effort enters the model (a generated assumption)
The data has no "study sessions". We **generate** the link: one session adds 100 clicks, 2 active days and
4 resources accessed to the student's activity features (`src/integration/effort_model.py`).
Why three features: when only clicks were raised, the NN lowered its prediction in **55% of steps**
(133 of 240 in `results/search_csp/monotonicity.csv`) because that input combination does not occur in real
data. With the three-feature rule there are **0 violations** (30 profiles x 8 steps).
The planner also applies a **monotone guard** (mark never decreases with more sessions) as a safety net.
Raw and adjusted marks are kept **separate**: `raw_predicted_marks` is the untouched NN output,
`planning_predicted_marks` (same as `predicted_marks`) is what Search used, and `monotone_adjusted` says per
assignment whether the guard changed the number. Corrections are also counted in `monotone_guard`.

**These numbers are assumed, not measured.** Nothing in OULAD says how much activity a study session adds or
what it does to a mark. Every plan therefore carries `effort_assumption` (`status: simulated_assumption`,
`measured_from_data: False`). Experiment 4 (`results/search_csp/effort_sensitivity.csv`) shows how much the
plan depends on the guess: with half the activity per session only 60% of plans are found and they need 12.3
sessions on average; with the default 7.4; with double 4.7. Treat "N sessions" as "N sessions *if* a session
adds this much activity".

## Other limits
* One course (FFF), 8 NN inputs (no assessment-type column, no 3 click groups). Small effect of effort:
  about +0.5 mark per session.
* The NN is modest. Same held-out rows, same metric code (`src/baselines.py`): NN MAE 9.7 / R² 0.35; training-mean
  baseline MAE 12.6 / R² -0.01; previous-TMA-mean baseline MAE 13.4 / R² -0.27 (`results/nn/evaluation_metrics.csv`).
* Targets used in experiments are **generated** (rules in `docs/data_dictionary.md`); timetables are simulated.
* Search with the monotone NN: if the cheapest NN-approved plan cannot be scheduled, every costlier plan
  cannot either, so the planner then reports `schedule_infeasible` and the best schedulable marks.
