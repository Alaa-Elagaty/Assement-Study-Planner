# Assessment-Aware Study Planner (Search + CSP + Neural Network)

A what-if planner for a distance-learning student: it finds the **fewest study sessions** that bring
predicted marks to a target, then builds a **week-by-week timetable**.

* **NN** (`src/nn`) predicts an assignment mark from earlier marks and online activity (OULAD, module FFF).
* **Search** (`src/search`) adds one session at a time: UCS, Greedy, A\*.
* **CSP** (`src/csp`) places sessions in weekly slots: Backtracking, Forward Checking, AC-3 (switchable).
* **Integration** (`src/integration`): Search proposes -> NN checks marks -> CSP checks the timetable.

## Setup
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # PyTorch is NOT needed to run anything below
```

## Run commands (from the project root)
| What | Command |
|---|---|
| Full demo + plots (`results/demo/`) | `python run_demo.py` (options: `--search ucs\|greedy\|astar`, `--csp bt\|fc\|ac3`) |
| All experiments, tables, plots (`results/search_csp/`) | `python -m src.eval.experiments` |
| All tests | `python -m unittest discover -s tests -t .` |
| Rebuild data from raw OULAD | `python make_data.py` (see below) |
| Retrain the NN (needs `pip install -r requirements-train.txt`) | `python make_data.py --train` |

Rebuilding data: download OULAD (CC BY 4.0, https://analyse.kmi.open.ac.uk/open_dataset) and unzip it into
`data/raw/oulad/`. The processed files, model, and scaler are already included, so this is optional.

## Project layout
`src/nn` network + predictor · `src/search` UCS/Greedy/A\* · `src/csp` BT/FC/AC-3 · `src/integration` planner + effort model ·
`src/eval` experiments · `tests/` · `docs/` (data dictionary, integration guide, limitations) · `results/` outputs.

## Important
This is a **decision-support what-if tool**. The link between study effort and marks is
observational, not proof of cause. Read `docs/limitations.md`.
