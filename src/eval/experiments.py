"""Reproducible experiments (seeded). Run:  python -m src.eval.experiments

Experiments
  1. UCS vs Greedy vs A*      : nodes expanded / generated, sessions, time
  2. BT vs BT+FC vs BT+FC+AC3 : 2, 3, 4 slots per week (seeded random instances)
  3. NN vs average-mark predictor inside the planner
  +  monotonicity check, and the planner on 30 student profiles
Every experiment uses SEEDS (10 seeds) and reports mean and sample std.
"""
import csv
import random
import statistics
from pathlib import Path

import pandas as pd

from src.integration.effort_model import FEATURES, adjust_features_for_effort
from src.integration.planner import StudyPlanner
from src.csp.domains import TimeSlot, build_domains
from src.csp.backtracking import solve_bt
from src.csp.forward_checking import solve_fc
from src.csp.ac3 import solve_ac3

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "search_csp"
PLOTS_DIR = RESULTS_DIR / "plots"
SEEDS = list(range(10))          # >= 5 seeds required
WEEKS, SLOTS_PER_WEEK = 6, 3
DEADLINES = [2, 4, 6]            # three assignments, staggered deadlines
FIXED_TARGET = 80.0              # common target mark used in experiment 3


def _write_csv(name, rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return path


def _mean_std(values):
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def reference_mark(f):
    """Mark the student 'usually gets': mean of previous TMAs (last score / 55 if none)."""
    if int(f["previous_tma_count"]) > 0:
        return float(f["previous_tma_mean"])
    return float(f["previous_tma_last_score"]) or 55.0


# ----------------------------------------------------------------- data / instances
def load_student_pool():
    """Held-out (later presentation) rows used as student/assignment profiles."""
    path = ROOT / "data" / "processed" / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run the data pipeline first: {path} not found")
    frame = pd.read_csv(path)
    missing = [f for f in FEATURES if f not in frame.columns]
    if missing:
        raise ValueError(f"test.csv is missing model features: {missing}")
    return frame


def make_student(pool, seed, n_assignments=3):
    """Seeded student: 3 assignment profiles (real rows) + 1-4 mark improvement goals."""
    rng = random.Random(seed)
    rows = pool.sample(n=n_assignments, random_state=seed)
    profiles = [{k: float(r[k]) for k in FEATURES} for _, r in rows.iterrows()]
    gains = [rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) for _ in range(n_assignments)]
    actual = [float(r["score"]) for _, r in rows.iterrows()]
    ks = [rng.randint(1, 5) for _ in range(n_assignments)]
    return {"profiles": profiles, "gains": gains, "actual": actual, "ks": ks}


def make_slots(slots_per_week=SLOTS_PER_WEEK, weeks=WEEKS):
    return [TimeSlot(w, p) for w in range(1, weeks + 1) for p in range(1, slots_per_week + 1)]


def build_assignments(student, predictor, target_rule="kmark"):
    """GENERATED target-mark rules (documented in docs/data_dictionary.md):
      'kmark': target = NN mark (monotone) after k sessions, k seeded in 1..5   -> reachable by design
      'fixed': every assignment has the same target mark FIXED_TARGET        -> predictor-independent
    """
    out = []
    for i, prof in enumerate(student["profiles"]):
        if target_rule == "kmark":
            target = max(predictor(adjust_features_for_effort(prof, n))["predicted_mark"]
                         for n in range(student["ks"][i] + 1))
        else:
            target = FIXED_TARGET
        out.append({"name": f"Assignment {i+1}", "target_mark": target,
                    "deadline_week": DEADLINES[i], "features": prof})
    return out


# ----------------------------------------------------------------- experiment 1
def run_search_comparison(pool, predictor):
    rows = []
    for seed in SEEDS:
        st = make_student(pool, seed)
        assigns = build_assignments(st, predictor)
        for algo in ("ucs", "greedy", "astar"):
            res = StudyPlanner(predictor, csp_algorithm="ac3").plan(assigns, make_slots(), algo)
            s = res["search_stats"]
            rows.append({"seed": seed, "algorithm": algo, "nodes_generated": s["nodes_generated"],
                         "nodes_expanded": s["nodes_expanded"], "session_cost": s["total_sessions"],
                         "execution_time_seconds": s["execution_time_seconds"], "feasible": res["success"],
                         "failure_type": res["failure_type"]})
    _write_csv("search_comparison.csv", rows)
    return rows


# ----------------------------------------------------------------- experiment 2
def make_csp_instance(seed, capacity, weeks=4, n_assignments=3):
    """Seeded CSP instance: 75-100% of the slots before the last deadline are demanded."""
    rng = random.Random(seed * 100 + capacity)
    deadlines = sorted(rng.sample(range(2, weeks + 1), k=min(n_assignments, weeks - 1)))
    while len(deadlines) < n_assignments:
        deadlines.append(weeks)
    rng.shuffle(deadlines)   # assignment order != deadline order, so naive BT must backtrack
    slots = [TimeSlot(w, p) for w in range(1, weeks + 1) for p in range(1, capacity + 1)]
    total = min(11, max(n_assignments, round(rng.uniform(0.75, 1.0) * weeks * capacity)))
    alloc = [1] * n_assignments
    while sum(alloc) < total:
        i = rng.randrange(n_assignments)
        if alloc[i] < capacity * deadlines[i]:
            alloc[i] += 1
    return alloc, slots, deadlines


def run_csp_comparison():
    rows = []
    for capacity in (2, 3, 4):
        for seed in SEEDS:
            alloc, slots, deadlines = make_csp_instance(seed, capacity)
            domains = build_domains(alloc, slots, deadlines)
            for name, solver in (("BT", solve_bt), ("BT+FC", solve_fc), ("BT+FC+AC3", solve_ac3)):
                r = solver(domains)
                st = r["stats"]
                rows.append({"slots_per_week": capacity, "seed": seed, "algorithm": name,
                             "sessions": sum(alloc), "assignments_attempted": st["assignments_attempted"],
                             "backtracks": st["backtracks"], "domain_values_removed": st["domain_values_removed"],
                             "constraint_checks": st["constraint_checks"],
                             "execution_time_seconds": st["execution_time_seconds"], "success": r["success"]})
    _write_csv("csp_comparison.csv", rows)
    return rows


# ----------------------------------------------------------------- experiment 3
def average_predictor(features):
    """Simple baseline: the student's average previous mark (ignores study effort)."""
    return {"predicted_mark": max(0.0, min(100.0, reference_mark(features)))}


def run_nn_baseline_comparison(pool, predictor):
    rows = []
    for seed in SEEDS:
        st = make_student(pool, seed)
        assigns = build_assignments(st, predictor, target_rule="fixed")   # same targets for both
        for label, model in (("neural_network", predictor), ("average_mark_baseline", average_predictor)):
            res = StudyPlanner(model, csp_algorithm="ac3").plan(assigns, make_slots(), "ucs")
            at0 = [model(p)["predicted_mark"] for p in st["profiles"]]
            err0 = statistics.mean(abs(a - b) for a, b in zip(at0, st["actual"]))
            # false reassurance: predictor says 'target already met' but the real score was below it
            false_ok = sum(1 for a, real in zip(at0, st["actual"]) if a >= FIXED_TARGET > real)
            rows.append({"seed": seed, "predictor": label, "feasible": res["success"],
                         "failure_type": res["failure_type"], "sessions": res["search_stats"]["total_sessions"],
                         "mean_abs_error_vs_actual_score": err0, "false_reassurance_count": false_ok})
    _write_csv("nn_vs_baseline.csv", rows)
    return rows


# ----------------------------------------------------------------- monotonicity
def run_monotonicity_experiment(pool, predictor, n_profiles=30, max_sessions=8):
    """Count steps where predicted mark DROPS when effort rises.

    raw_new = NN with the new effort rule (clicks + active days + resources)
    raw_old = NN with the old rule (clicks only)       guarded = what the planner uses
    """
    rows, tot = [], {"raw_new": 0, "raw_old": 0, "guarded": 0, "steps": 0}
    sample = pool.sample(n=n_profiles, random_state=0)
    for pid, (_, r) in enumerate(sample.iterrows()):
        f = {k: float(r[k]) for k in FEATURES}
        new = [predictor(adjust_features_for_effort(f, n))["predicted_mark"] for n in range(max_sessions + 1)]
        old = [predictor(adjust_features_for_effort(f, n, 100, 0, 0))["predicted_mark"] for n in range(max_sessions + 1)]
        guarded = [max(new[:i + 1]) for i in range(len(new))]
        for n in range(max_sessions + 1):
            rows.append({"profile_id": pid, "sessions": n, "raw_old_rule": old[n], "raw_new_rule": new[n], "guarded": guarded[n]})
        for seq, key in ((new, "raw_new"), (old, "raw_old"), (guarded, "guarded")):
            tot[key] += sum(seq[i + 1] < seq[i] for i in range(max_sessions))
        tot["steps"] += max_sessions
    _write_csv("monotonicity.csv", rows)
    tot["profiles"] = n_profiles
    return {"rows": rows, **tot}


# ----------------------------------------------------------------- 30 profiles
def run_thirty_profiles(pool, predictor, n=30):
    rows = []
    for seed in range(n):
        st = make_student(pool, 1000 + seed)
        assigns = build_assignments(st, predictor)
        cap = (1, 2, 3)[seed % 3]            # students have 1, 2 or 3 free slots per week
        res = StudyPlanner(predictor).plan(assigns, make_slots(cap), "ucs")
        tt = ""
        if res["success"]:
            tt = "; ".join(f"W{w}: " + ",".join(x["session"] for x in items) for w, items in sorted(res["weekly_timetable"].items()))
        rows.append({"student": seed + 1, "slots_per_week": cap, "targets": res["targets"], "success": res["success"],
                     "failure_type": res["failure_type"], "sessions_per_assignment": res["allocation"],
                     "predicted_marks": res["predicted_marks"] or res["best_schedulable_marks"],
                     "total_sessions": res["search_stats"]["total_sessions"], "timetable": tt})
    _write_csv("planner_30_profiles.csv", rows)
    return rows


# ----------------------------------------------------------------- summary + plots
def write_summary(search, csp, nn, mono):
    summary = []

    def add(exp, group, metric, vals):
        m, s, n = _mean_std(vals)
        if n:
            summary.append({"experiment": exp, "group": group, "metric": metric, "mean": m, "std": s, "n": n})

    for a in ("ucs", "greedy", "astar"):
        sub = [r for r in search if r["algorithm"] == a]
        for metric in ("nodes_expanded", "nodes_generated", "session_cost", "execution_time_seconds"):
            add("1_search", a, metric, [r[metric] for r in sub])
        add("1_search", a, "feasible_rate", [int(r["feasible"]) for r in sub])
    for c in (2, 3, 4):
        for a in ("BT", "BT+FC", "BT+FC+AC3"):
            sub = [r for r in csp if r["slots_per_week"] == c and r["algorithm"] == a]
            for metric in ("assignments_attempted", "backtracks", "constraint_checks", "execution_time_seconds"):
                add("2_csp", f"{a}|{c}_slots", metric, [r[metric] for r in sub])
            add("2_csp", f"{a}|{c}_slots", "success_rate", [int(r["success"]) for r in sub])
    for p in ("neural_network", "average_mark_baseline"):
        sub = [r for r in nn if r["predictor"] == p]
        add("3_nn_vs_baseline", p, "feasible_rate", [int(r["feasible"]) for r in sub])
        add("3_nn_vs_baseline", p, "sessions_when_feasible", [r["sessions"] for r in sub])
        add("3_nn_vs_baseline", p, "mean_abs_error_vs_actual_score", [r["mean_abs_error_vs_actual_score"] for r in sub])
        add("3_nn_vs_baseline", p, "false_reassurance_count", [r["false_reassurance_count"] for r in sub])
    steps = mono["steps"]
    for k, label in (("raw_old", "clicks_only_rule"), ("raw_new", "new_effort_rule"), ("guarded", "planner_guarded")):
        add("monotonicity", label, "violation_rate", [mono[k] / steps])
        add("monotonicity", label, "violations", [mono[k]])
    _write_csv("experiment_summary.csv", summary)
    return summary


def make_plots(summary, mono):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    S = pd.DataFrame(summary)

    def bars(ax, exp, metric, groups, title, ylabel, log=False):
        d = S[(S.experiment == exp) & (S.metric == metric) & (S.group.isin(groups))].set_index("group").reindex(groups)
        lower = [min(sd, mu) for mu, sd in zip(d["mean"], d["std"])]
        ax.bar(range(len(groups)), d["mean"], yerr=[lower, list(d["std"])], capsize=4, color="#4c72b0")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=15)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("symlog")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    g = ["ucs", "greedy", "astar"]
    bars(ax[0], "1_search", "nodes_expanded", g, "Nodes expanded", "mean ± std (10 seeds)")
    bars(ax[1], "1_search", "nodes_generated", g, "Nodes generated", "")
    bars(ax[2], "1_search", "session_cost", g, "Total sessions (plan cost)", "")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "exp1_search.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, (metric, title) in enumerate((("backtracks", "Backtracks"), ("assignments_attempted", "Assignments attempted"))):
        w = 0.27
        for j, a in enumerate(("BT", "BT+FC", "BT+FC+AC3")):
            d = S[(S.experiment == "2_csp") & (S.metric == metric) & (S.group.str.startswith(a + "|"))]
            d = d.assign(c=d.group.str.extract(r"\|(\d)_")[0].astype(int)).sort_values("c")
            lower = [min(sd, mu) for mu, sd in zip(d["mean"], d["std"])]   # no bars below zero
            ax[k].bar([i + (j - 1) * w for i in range(len(d))], d["mean"], w, yerr=[lower, list(d["std"])], capsize=3, label=a)
        ax[k].set_xticks(range(3)); ax[k].set_xticklabels(["2 slots/wk", "3 slots/wk", "4 slots/wk"])
        ax[k].set_yscale("symlog", linthresh=1); ax[k].set_ylim(bottom=0); ax[k].set_title(title + " (log scale)"); ax[k].legend()
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "exp2_csp.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    g = ["neural_network", "average_mark_baseline"]
    bars(ax[0], "3_nn_vs_baseline", "feasible_rate", g, "Plans found (rate)", "")
    bars(ax[1], "3_nn_vs_baseline", "sessions_when_feasible", g, "Sessions when a plan exists", "")
    bars(ax[2], "3_nn_vs_baseline", "mean_abs_error_vs_actual_score", g, "Mark error vs real score", "marks")
    bars(ax[3], "3_nn_vs_baseline", "false_reassurance_count", g, "False 'target met' (of 3)", "")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "exp3_nn_vs_baseline.png", dpi=130); plt.close(fig)

    m = pd.DataFrame(mono["rows"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for pid in range(6):
        d = m[m.profile_id == pid]
        ax.plot(d.sessions, d.raw_old_rule, "--", alpha=.6, color=f"C{pid}")
        ax.plot(d.sessions, d.raw_new_rule, "-", color=f"C{pid}")
    ax.set_xlabel("extra sessions"); ax.set_ylabel("predicted mark")
    ax.set_title("Monotonicity: dashed = clicks-only rule, solid = new effort rule")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "monotonicity.png", dpi=130); plt.close(fig)


def run_all_experiments():
    from src.nn.predictor import predict_mark
    pool = load_student_pool()
    search = run_search_comparison(pool, predict_mark)
    csp = run_csp_comparison()
    nn = run_nn_baseline_comparison(pool, predict_mark)
    mono = run_monotonicity_experiment(pool, predict_mark)
    profiles = run_thirty_profiles(pool, predict_mark)
    summary = write_summary(search, csp, nn, mono)
    make_plots(summary, mono)
    return {"summary": summary, "monotonicity": {k: v for k, v in mono.items() if k != "rows"},
            "profiles_planned": len(profiles), "results_directory": str(RESULTS_DIR)}


if __name__ == "__main__":
    import time
    t = time.time()
    out = run_all_experiments()
    print(pd.DataFrame(out["summary"]).round(4).to_string(index=False))
    print(out["monotonicity"], "profiles:", out["profiles_planned"], f"({time.time()-t:.1f}s)")
