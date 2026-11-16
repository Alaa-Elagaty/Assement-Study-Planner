# FFF TMA Prediction – Data Dictionary

## Target Variable

| Column | Description |
|---|---|
| score | Actual student's TMA mark (0–100). This is the value predicted by the neural network. |

## Model Input Features

| Feature | Description |
|---|---|
| date | Assessment deadline, measured in days from the course start. |
| weight | Percentage contribution of the assessment to the overall assessment score. |
| previous_tma_count | Number of previous TMAs completed by the student in the same presentation. |
| previous_tma_mean | Average score across the student's previous TMAs. |
| previous_tma_last_score | Score obtained in the student's most recent previous TMA. |
| previous_clicks | Total VLE clicks before the current assessment deadline. |
| previous_active_days | Number of distinct days with VLE activity before the deadline. |
| previous_resources_accessed | Number of distinct VLE resources accessed before the deadline. |

## Dataset Information

- Dataset: Open University Learning Analytics Dataset (OULAD).
- Selected module: FFF.
- Presentations: 2013B, 2013J, 2014B, 2014J.
- Target: TMA assessment score.
- Total records: 24,777.
- Training records: 12,881.
- Validation records: 4,636.
- Testing records: 7,260.

## Data Splitting

- Training: 2013B and 2013J.
- Validation: 2014B.
- Testing: 2014J.

## Preprocessing

- Missing target scores were excluded.
- Historical features use only information available before the target assessment.
- VLE activity was filtered to dates before the target deadline.
- Numerical input features were standardized using StandardScaler.
- The scaler was fitted only on the training dataset.

## Excluded Information

Student demographic information and final_result were excluded from model inputs. The final_result column was excluded to avoid target leakage.

## Important Notes

- The model predicts an estimated mark, not a guaranteed result.
- Valid assessment marks range from 0 to 100.
- The neural network may produce raw predictions outside this range. The predictor will bound displayed predictions to 0–100.
- Input features must be supplied in the exact order expected by the model.

## Generated (NOT real) quantities used by the planner
| Name | Value / rule | Where |
|---|---|---|
| Session length | 1 session = +100 clicks, +2 active days, +4 resources accessed | `src/integration/effort_model.py` |
| Slots per week | 1-3 (experiments use 3; 30-profile run uses 1, 2 or 3) | `src/eval/experiments.py` |
| Deadlines | assignment 1/2/3 due weeks 2/4/6 (inclusive) | `src/eval/experiments.py` |
| Target mark, rule `kmark` | NN mark after k sessions, k drawn 1-5 per assignment (seeded) | experiments 1, 30 profiles |
| Target mark, rule `fixed` | 80 for every assignment | experiment 3 |
| CSP instances | seeded random: 3 assignments, 75-100% of slots demanded, shuffled deadlines | experiment 2 |
| Seeds | 0-9 (10 seeds); 30 profiles use seeds 1000-1029 | `SEEDS` |

Real: scores, clicks, active days, resources (OULAD). Student profiles for planning are real rows of the
held-out presentation 2014J (`data/processed/test.csv`).
