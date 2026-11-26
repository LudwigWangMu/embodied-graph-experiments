"""Check shipped file hashes and reconstruct reported results from observation-level files."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def rows(study, name):
    with (ROOT / study / "data" / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def check(same, label):
    if not same:
        raise ValueError(label)


def near(actual, expected, label):
    check(math.isclose(actual, float(expected), abs_tol=1e-9, rel_tol=1e-10), label)


def contact(study):
    groups = defaultdict(dict)
    for r in rows(study, "branch_pairs.csv"):
        if int(r["step"]) == 16:
            key = (r["shape"], r["engine"], r["view"], r["model"])
            check(r["scene"] not in groups[key], key)
            groups[key][r["scene"]] = float(r["passive_mae_mm"])
    summaries = rows(study, "branch_summary.csv")
    check(len(summaries) == 8, "fork/packing comparison groups")
    for r in summaries:
        design = (r["shape"], r["engine"], r["view"])
        for model in ["Zero", "Graph", "Events", "Corrected"]:
            values = groups[(*design, model)]
            check(len(values) == 240, design)
            near(mean(values.values()), r[model.lower()], (design, model))
        corrected, events = (groups[(*design, m)] for m in ["Corrected", "Events"])
        check(corrected.keys() == events.keys(), design)
        near(mean(corrected[k] - events[k] for k in events), r["paired_delta"], design)
    print("contact-response: eight paired 240-scene comparisons agree")


def prefixes(study):
    for stem, section, violations in [("frozen_vla", "study", "violations"),
                                      ("visual_diffusion", "results", "violating")]:
        summary = json.loads((ROOT / study / "data" / (stem + "_summary.json")).read_text())
        episodes = rows(study, stem + "_episodes.csv")
        for mode, reported in summary[section].items():
            runs = [r for r in episodes if r["mode"] == mode]
            check(len({r["seed"] for r in runs}) == len(runs) == 24, mode)
            for col in ["success", "steps", "calls"]:
                near(sum(int(r[col]) for r in runs), reported[col], (stem, mode, col))
            near(sum(int(r["authorized_violation"]) for r in runs), reported[violations], mode)
            near(mean(float(r["max_coverage"]) for r in runs), reported["peak_overlap"], mode)
    longer = rows(study, "visual_diffusion_horizon.csv")
    check(len(longer) == 24 and sum(int(r["success"]) for r in longer) == 16, "300-action horizon")
    print("action-prefixes: two frozen-policy summaries and 300-action horizon agree")


def readouts(study):
    packets = rows(study, "shared_slot_packets.csv")
    for r in rows(study, "shared_slot_summary.csv"):
        selected = [p for p in packets if p["rule"] == r["rule"]]
        near(len(selected), r["arrivals"], r["rule"])
        near(sum(int(p["wrong_winner"]) for p in selected), r["wrong_winner"], r["rule"])
        near(sum(float(p["output_distance"]) > 1e-10 for p in selected), r["numeric_mismatch"], r["rule"])
    directed = rows(study, "directed_release.csv")
    stale = sum(int(p["accepted"]) and float(p["oracle_delta"]) > 1e-10 for p in directed)
    summary = json.loads((ROOT / study / "data/queued_summary.json").read_text())
    near(len(directed), summary["directed_tests"], "directed release count")
    near(stale, summary["accepted_stale"], "accepted stale count")
    print("graph-readouts: release and ordering summaries agree")


def sensing(study):
    risks = rows(study, "policy_risks.csv")
    for r in rows(study, "summary.csv"):
        if r["method"] != "learned":
            values = [float(p["risk"]) for p in risks if p["split"] == r["split"]
                      and p["method"] == r["method"] and float(p["noise"]) == 0.1]
            near(mean(values), r["risk"], (r["split"], r["method"]))
    routes = rows(study, "navigation.csv")
    for r in rows(study, "route_summary.csv"):
        group = [p for p in routes if p["split"] == r["split"] and p["method"] == r["method"]]
        near(len(group), r["runs"], "route count")
        excess = [int(p["extra_actions"]) + 2 * int(p["bumps"]) + 0.1 * int(p["queries"]) for p in group]
        near(mean(excess), r["charged_excess"], r["method"])
    print("active-frame-sensing: one-step risks and 5,760 route costs agree")


STUDIES = {"contact-response": contact, "action-prefixes": prefixes,
           "graph-readouts": readouts, "active-frame-sensing": sensing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=STUDIES)
    parser.add_argument("--results-only", action="store_true",
                        help="Check numerical results after a rerun without comparing original file bytes.")
    args = parser.parse_args()
    if not args.results_only:
        for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            if not args.study or name.startswith(args.study + "/"):
                check(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name)
    for name, compute in STUDIES.items():
        if args.study is None or name == args.study:
            compute(name)
