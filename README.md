# Assessment-Aware Study Planner

A study-planning tool that combines a neural network, search algorithms, and constraint satisfaction to build a weekly study timetable around target marks and assignment deadlines.

The planner estimates how many additional study sessions each assignment may need, then checks whether those sessions can fit into the available time slots. It is designed as a decision-support and experimentation project, not as a guarantee of academic results.

## How It Works

The system has four main components:

1. **Neural Network (`src/nn/`)** — estimates a TMA mark using a student's previous assessment performance and VLE activity.
2. **Search Algorithms (`src/search/`)** — explores possible study-session allocations using Uniform Cost Search (UCS), Greedy Best-First Search, or A*.
3. **Constraint Satisfaction (`src/csp/`)** — checks whether the proposed sessions can be placed into available weekly slots before their deadlines. It supports Backtracking, Forward Checking, and AC-3.
4. **Integration (`src/integration/`)** — connects the predictor, search process, and timetable solver.

Search proposes an allocation, the neural network evaluates it, and the CSP checks whether it can be scheduled.

A plan is accepted only when its predicted marks meet the requested targets and the timetable constraints can be satisfied.

## Key Features

- Predicts TMA marks using eight academic-performance and online-activity features.
- Compares UCS, Greedy Best-First Search, and A* for allocating study sessions.
- Compares Backtracking, Forward Checking, and AC-3 for timetable feasibility.
- Respects assignment deadlines and prevents two sessions from using the same time slot.
- Identifies why a plan could not be generated.
- Keeps raw neural-network predictions separate from the adjusted values used during planning.
- Includes experiments, evaluation metrics, result tables, and visualizations.

## Project Structure

```text
.
├── data/
│   └── processed/
├── docs/
│   ├── data_dictionary.md
│   ├── integration_guide.md
│   ├── limitations.md
│   └── results_analysis.md
├── models/
├── results/
│   ├── demo/
│   ├── nn/
│   └── search_csp/
├── src/
│   ├── csp/
│   ├── eval/
│   ├── integration/
│   ├── nn/
│   └── search/
├── tests/
├── requirements.txt
├── requirements-train.txt
└── run_demo.py
```

## Getting Started

### Requirements

- Python 3.12
- pip
- Project files, including the trained model and scaler in `models/`

PyTorch is optional for normal prediction and planning because the project includes exported NumPy weights. It is required for retraining the neural network.

### Installation

**1. Clone the repository**

```bash
git clone <repository-url>
cd <repository-folder>
```

Replace the placeholders with your repository URL and folder name.

**2. Create a virtual environment**

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the following commands from the project root.

## Running the Project

### Run the Demo

```bash
python run_demo.py
```

The demo runs a sample planning scenario, displays the resulting allocation and timetable, and saves an overview plot in `results/demo/`.

You can select different search and CSP algorithms:

```bash
python run_demo.py --search ucs --csp ac3
python run_demo.py --search astar --csp fc
python run_demo.py --search greedy --csp bt
```

Available search algorithms:

- `ucs` — Uniform Cost Search
- `greedy` — Greedy Best-First Search
- `astar` — A* Search

Available CSP algorithms:

- `bt` — Backtracking
- `fc` — Forward Checking
- `ac3` — Arc Consistency 3

To run without generating plots:

```bash
python run_demo.py --no-plots
```

### Run Experiments

```bash
python -m src.eval.experiments
```

The experiments compare:

- Search algorithm performance.
- CSP solver performance.
- Neural-network predictions against baseline predictors.
- Sensitivity to the assumed activity added by each study session.
- Prediction monotonicity across increasing session counts.
- Planning performance across 30 student profiles.

Results are saved in `results/search_csp/`.

### Run Tests

```bash
python -m unittest discover -s tests -t .
```

### Rebuild the Dataset

The processed data, trained weights, and scaler are included, so rebuilding the dataset is optional.

If the required raw OULAD files are unavailable, the pipeline attempts to download the dataset.

```bash
python -m src.make_data
```

The pipeline prepares target data, creates historical features, adds VLE activity features, generates train/validation/test splits, and fits the scaler using the training split.

The dataset is divided by presentation:

| Split | Presentation |
|---|---|
| Training | 2013B and 2013J |
| Validation | 2014B |
| Testing | 2014J |

To retrain the neural network:

```bash
python -m pip install -r requirements-train.txt
python -m src.make_data --train
```

Retraining requires PyTorch.

## Using the Neural Network Predictor

The prediction function is available in `src/nn/predictor.py`.

```python
from src.nn.predictor import predict_mark

student_features = {
    "date": 47,
    "weight": 12.5,
    "previous_tma_count": 1,
    "previous_tma_mean": 78,
    "previous_tma_last_score": 78,
    "previous_clicks": 1200,
    "previous_active_days": 15,
    "previous_resources_accessed": 60,
}

result = predict_mark(student_features)

print(result["predicted_mark"])
```

All eight features must be provided using their exact names.

The predictor handles feature scaling internally, so manual scaling is unnecessary.

The returned dictionary contains:

- `raw_prediction` — the original neural-network output.
- `predicted_mark` — the prediction bounded between 0 and 100.

For more details, see [`docs/integration_guide.md`](docs/integration_guide.md).

## Model and Planning Details

The neural network uses eight input features:

- Assessment date.
- Assessment weight.
- Number of previous TMAs.
- Mean previous TMA score.
- Most recent previous TMA score.
- Previous VLE clicks.
- Previous active days.
- Previous resources accessed.

The planner does not add an arbitrary mark bonus when another study session is scheduled.

Instead, `src/integration/effort_model.py` adjusts the student's activity features before requesting a new prediction.

By default, one simulated study session adds:

| Activity Feature | Added Per Session |
|---|---:|
| VLE clicks | 100 |
| Active days | 2 |
| Resources accessed | 4 |

**Important:** These values are assumptions, not measurements from OULAD. The dataset does not record study sessions, and the observed relationship between online activity and marks does not establish that additional activity causes higher marks.

The planner also uses a monotone guard to prevent the marks used during planning from decreasing as the number of simulated sessions increases.

Raw neural-network outputs are stored separately from adjusted planning predictions.

See [`docs/limitations.md`](docs/limitations.md) before interpreting the results.

## Evaluation Results

The repository includes saved evaluation metrics and experiment outputs.

### Neural Network Performance

On the full held-out test set of 7,260 records:

| Model | MAE |
|---|---:|
| Neural Network | 9.7 |
| Training-Mean Baseline | 12.6 |
| Previous-TMA-Mean Baseline | 13.4 |

### Search Algorithm Comparison

In the 10-seed experiment:

- All three search algorithms found plans in all tested cases.
- Their mean session cost was 7.4.
- A* expanded fewer nodes on average than UCS.

### CSP Performance

The experiments showed that AC-3 substantially reduced backtracking on the tested tight scheduling instances compared with plain Backtracking and Forward Checking.

These results describe the project's experiments and should not be treated as guarantees of performance on other datasets or students.

For detailed results and visualizations, see [`docs/results_analysis.md`](docs/results_analysis.md).

## Documentation

| Document | Description |
|---|---|
| [Data Dictionary](docs/data_dictionary.md) | Dataset, target variable, features, and preprocessing |
| [Integration Guide](docs/integration_guide.md) | Neural-network integration and prediction interface |
| [Results Analysis](docs/results_analysis.md) | Model evaluation and search/CSP experiments |
| [Limitations](docs/limitations.md) | Assumptions, interpretation, and known limitations |

## Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**, specifically TMA records from the FFF module.

The prepared dataset contains 24,777 records.

- Dataset: [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- Dataset License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Project Scope

This project is an academic decision-support prototype demonstrating how machine learning, search algorithms, and constraint satisfaction can work together in a study-planning system.

Its predictions depend on the available FFF data and an explicitly simulated relationship between study sessions and VLE activity.

The planner is intended to explore possible study schedules, not to guarantee academic marks or replace academic guidance.
