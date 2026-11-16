"""Integrated demo:  python run_demo.py   (options: --search ucs|greedy|astar  --csp bt|fc|ac3)

Shows (1) the input, (2) the plan + weekly timetable, (3) plots of how the NN, Search
and CSP cooperate, (4) the case "NN says enough but CSP rejects", (5) target unreachable,
(6) NN ON vs OFF, and (7) CSP pruning with and without FC / AC-3.
Plots are saved in results/demo/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.nn.predictor import predict_mark
from src.integration.effort_model import adjust_features_for_effort
from src.integration.planner import StudyPlanner, monotone_envelope
from src.csp.domains import TimeSlot, build_domains
from src.csp import solve_bt, solve_fc, solve_ac3
from src.eval import experiments as E

OUT = Path(__file__).resolve().parent / "results" / "demo"
COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def banner(text):
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72)


def show_input(assigns, slots_per_week):
    print(f"Free study slots: {slots_per_week} per week, weeks 1-{E.WEEKS}")
    for a in assigns:
        f = a["features"]
        print(f"  {a['name']}: deadline week {a['deadline_week']}, target {a['target_mark']:.2f} | "
              f"previous TMAs={int(f['previous_tma_count'])}, mean={f['previous_tma_mean']:.0f}, clicks={int(f['previous_clicks'])}")


def show_result(res):
    print("Search:", res["search_algorithm"].upper(), "| CSP:", res["csp_algorithm"].upper())
    print("Predicted marks with NO extra study:", res["marks_with_no_extra_study"])
    st = res["search_stats"]
    print(f"Search counters: expanded={st['nodes_expanded']} generated={st['nodes_generated']}")
    ct = res["csp_total_stats"]
    print(f"CSP counters   : assignments={ct['assignments_attempted']} backtracks={ct['backtracks']}")
    print("Monotone guard :", res["monotone_guard"])
    print("Effort assumption:", res["effort_assumption"]["per_session"], "per session (simulated, NOT measured)")
    if res["success"]:
        print(f"PLAN FOUND  sessions per assignment = {res['allocation']}  (total {sum(res['allocation'])})")
        print("Planning marks (monotone-adjusted):", res["predicted_marks"])
        print("Raw NN marks (unadjusted)         :", res["raw_predicted_marks"], "adjusted?", res["monotone_adjusted"])
        for week, items in sorted(res["weekly_timetable"].items()):
            print(f"  Week {week}: " + ", ".join(f"slot {x['slot']} -> {x['session']}" for x in items))
    else:
        print("NO PLAN  [" + res["failure_type"] + "]")
        print(" ", res["failure_reason"])
        print("  Best marks if slots were unlimited (not schedulable):", res["best_predicted_marks"], "at", res["best_allocation"])
        print("  Best marks that can really be SCHEDULED            :", res["best_schedulable_marks"], "at", res["best_schedulable_allocation"])


def plot_overview(res, assigns, predictor, slots_per_week, csp_rows, path):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    # 1. NN: predicted mark vs sessions
    for i, a in enumerate(assigns):
        xs = list(range(0, 11))
        raw = [predictor(adjust_features_for_effort(a["features"], n))["predicted_mark"] for n in xs]
        ys = monotone_envelope(raw)                      # planning marks (what search uses)
        ax[0, 0].plot(xs, raw, color=COLORS[i], ls="--", alpha=.5, lw=1)   # raw NN output, never edited
        ax[0, 0].plot(xs, ys, color=COLORS[i], label=a["name"])
        ax[0, 0].axhline(a["target_mark"], color=COLORS[i], ls=":")
        if res["allocation"]:
            n = res["allocation"][i]
            ax[0, 0].scatter([n], [res["predicted_marks"][i]], color=COLORS[i], s=90, edgecolor="k", zorder=3)
    ax[0, 0].set_title("1. NN mark vs sessions (solid = planning, dashed = raw NN, dotted = target)")
    ax[0, 0].set_xlabel("sessions"); ax[0, 0].set_ylabel("predicted mark"); ax[0, 0].legend()
    # 2. Search trace
    tr = res["search_trace"]
    xs = [t["order"] for t in tr]
    ax[0, 1].plot(xs, [t["gap"] for t in tr], color="#999", lw=1, label="mark gap to target")
    for t in tr:
        if t["is_goal"]:
            ax[0, 1].scatter(t["order"], t["gap"], s=110, marker="*" if t["csp_ok"] else "X",
                             color="green" if t["csp_ok"] else "red", zorder=3)
    ax[0, 1].scatter([], [], marker="*", color="green", s=110, label="goal accepted by CSP")
    ax[0, 1].scatter([], [], marker="X", color="red", s=110, label="goal rejected by CSP")
    ax[0, 1].set_title(f"2. Search trace ({res['search_algorithm'].upper()}): order nodes were expanded")
    ax[0, 1].set_xlabel("expansion order"); ax[0, 1].set_ylabel("total mark gap"); ax[0, 1].legend()
    # 3. Timetable
    ax[1, 0].set_title("3. CSP timetable (week x slot)")
    ax[1, 0].set_xlim(0.5, E.WEEKS + 0.5); ax[1, 0].set_ylim(slots_per_week + 0.5, 0.5)
    ax[1, 0].set_xlabel("week"); ax[1, 0].set_ylabel("slot"); ax[1, 0].set_xticks(range(1, E.WEEKS + 1))
    ax[1, 0].set_yticks(range(1, slots_per_week + 1))
    for a in assigns:
        ax[1, 0].axvline(a["deadline_week"] + 0.5, color="k", ls="--", lw=.7)
    if res["timetable"]:
        for var, it in res["timetable"].items():
            i = int(var.split("_")[0][1:]) - 1
            ax[1, 0].add_patch(plt.Rectangle((it["week"] - .45, it["position"] - .4), .9, .8, color=COLORS[i]))
            ax[1, 0].text(it["week"], it["position"], var.replace("_", "·"), ha="center", va="center", color="w", fontsize=8)
    else:
        ax[1, 0].text(0.5, 0.5, "no feasible timetable", transform=ax[1, 0].transAxes, ha="center")
    # 4. CSP pruning
    names = ["BT", "BT+FC", "BT+FC+AC3"]
    w = .38
    ax[1, 1].bar([i - w / 2 for i in range(3)], [csp_rows[n]["backtracks"] for n in names], w, label="backtracks")
    ax[1, 1].bar([i + w / 2 for i in range(3)], [csp_rows[n]["assignments_attempted"] for n in names], w, label="assignments tried")
    ax[1, 1].set_xticks(range(3)); ax[1, 1].set_xticklabels(names); ax[1, 1].set_yscale("symlog", linthresh=1)
    ax[1, 1].set_ylim(0, 5 * max(csp_rows[n]["assignments_attempted"] for n in names))
    ax[1, 1].set_title("4. CSP pruning on one tight instance (with / without FC, AC-3)"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", choices=["ucs", "greedy", "astar"], default="ucs")
    ap.add_argument("--csp", choices=["bt", "fc", "ac3"], default="ac3")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pool = E.load_student_pool()
    student = E.make_student(pool, 3)
    assigns = E.build_assignments(student, predict_mark)
    spw = 3
    slots = E.make_slots(spw)

    banner("1. INPUT: one student, three assignments (real held-out OULAD profiles)")
    show_input(assigns, spw)
    banner("2. PLAN: Search proposes -> NN checks marks -> CSP checks the timetable")
    res = StudyPlanner(predict_mark, csp_algorithm=args.csp).plan(assigns, slots, args.search)
    show_result(res)

    # CSP pruning comparison on one tight instance (same instance, three solvers)
    alloc, cslots, dls = E.make_csp_instance(3, 3)
    dom = build_domains(alloc, cslots, dls)
    csp_rows = {n: s(dom)["stats"] for n, s in (("BT", solve_bt), ("BT+FC", solve_fc), ("BT+FC+AC3", solve_ac3))}
    banner("3. CSP PRUNING: same tight instance, three solvers")
    print(f"Instance: sessions per assignment={alloc}, deadlines (weeks)={dls}, 3 slots/week")
    for n, st in csp_rows.items():
        print(f"  {n:<10} assignments tried={st['assignments_attempted']:>6}  backtracks={st['backtracks']:>5}  removed values={st['domain_values_removed']}")
    if not args.no_plots:
        p = OUT / "demo_overview.png"
        plot_overview(res, assigns, predict_mark, spw, csp_rows, p)
        print("\nPlot saved:", p)

    banner("4. NN says 'enough' but the CSP REJECTS the schedule")
    # Two assignments both due in week 1 with ONE slot: each needs 1 session, but they cannot share the slot.
    two = [dict(a, deadline_week=1) for a in assigns[:2]]
    for a in two:
        a["target_mark"] = max(predict_mark(adjust_features_for_effort(a["features"], n))["predicted_mark"] for n in range(2))
    rej = StudyPlanner(predict_mark, csp_algorithm=args.csp).plan(two, [TimeSlot(1, 1)], args.search)
    print("NN-approved allocations that were rejected by the CSP:", [g["allocation"] for g in rej["rejected_goals"]])
    show_result(rej)
    for n, s in (("BT", solve_bt), ("BT+FC", solve_fc), ("BT+FC+AC3", solve_ac3)):
        r = s(build_domains([1, 1], [TimeSlot(1, 1)], [1, 1]))
        print(f"  {n:<10} on that plan -> success={r['success']} (backtracks={r['stats']['backtracks']})")

    banner("5. TARGET UNREACHABLE at this capacity")
    hard = [dict(a, target_mark=99.0) for a in assigns]
    show_result(StudyPlanner(predict_mark, csp_algorithm=args.csp).plan(hard, E.make_slots(2), args.search))

    banner("6. SYSTEM WITH NN ON vs OFF (average-mark predictor)")
    off = StudyPlanner(E.average_predictor, csp_algorithm=args.csp).plan(assigns, slots, args.search)
    print("NN ON : plan found =", res["success"], "| sessions =", res["allocation"])
    print("NN OFF: plan found =", off["success"], "|", off["failure_type"] or off["allocation"])
    print("(NN OFF uses the shared previous-mark baseline from src/baselines.py)")
    print("The average-mark predictor ignores study effort, so it can never say that more sessions help.")
    print("\nNOTE: this is a what-if tool. Links between effort and marks are observational, not causal.")


if __name__ == "__main__":
    main()
