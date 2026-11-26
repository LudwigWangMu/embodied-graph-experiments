# Which Way Should a Robot Look? Decision-Relevant Symmetry Breaking on Navigation Graphs

## Reproduce the study

The numerical work was run on CPU with Python 3.12.15. The exact package versions used are recorded in `requirements.txt`. In an environment with those dependencies installed, run the following commands from the package directory.

```text
python -m pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` rebuilds the maps and query targets, evaluates exact policies, trains graph and dense selectors with three fixed seeds each, simulates the routes, tests the handedness prior, regenerates the four vector figures, and verifies the output tables. The ten upstream map archives are included for offline computation. Initial dependency installation may require internet access. For a shorter check using the distributed data, run `python code/verify.py`. A different PyTorch release or processor may give small floating-point differences during training.

## Find a reported number

| Manuscript item | Data or calculation |
| --- | --- |
| Sampling, split counts, origin, goal, distance | `data/tasks.csv`; `data/tasks.npz` stores the free-cell masks, goals, frame-by-command losses and probe-by-frame signatures. |
| One-step outcomes and paired intervals, Table 2 and Figure 2 | `data/policy_risks.csv`, `data/learned_risks.csv`, `data/summary.csv`, `data/paired_intervals.csv` |
| Example occupancy and goal in Figure 1 | `data/checks.json` identifies the selected task; `code/report.py` draws it from its saved arrays. |
| Signature counts, full-footprint bound, and bit errors, Table 3 and Figure 3 | `data/ambiguity.csv`, `data/noise_sweep.csv` |
| Reflection prior and dependent sensor noise | `data/handedness.csv`, `data/sensor_dependence.csv` |
| Per-run route outcomes, Table 4 and Figure 4 | `data/navigation.csv`, `data/route_summary.csv`, `data/route_intervals.csv` |
| Action-entropy and parameter-matched dense controls | `data/comparison_controls.csv`, `code/comparators.py`, three `data/dense_probe_*.pt` checkpoints |
| Graph model and fitting targets | `code/learn.py`, `data/graph_training.npz`, `data/validation_loss.csv`, three `data/probe_net_*.pt` checkpoints |
| Integrity, exact-search and relabeling checks | `data/verification.json`, `../SHA256SUMS.txt` |

Risk integrates over frame hypotheses, answer noise and specified ties. Route outcomes are seeded trajectories with exact state feedback. The ten-city group is the unit resampled for street-map intervals. Each connected closure shares its starting task with a procedural test map. The eight disconnected closures are excluded from finite-distance comparisons.

## Map attribution

This work contains information from the Moving AI grid benchmark street collection, made available under the Open Data Commons Attribution License v1.0. Konstantin Yakovlev and Anton Andreychuk provided the discretized street maps. The ten original archives are included in `data/maps/`. The initial crop, reachability filter, goal choice, four-neighbor routes and simulated observations were generated for this study. `data/map_sources.csv` records a URL and SHA-256 hashes for every original archive and map. See N. R. Sturtevant, *Benchmarks for Grid-Based Pathfinding* (2012). [Street collection](https://movingai.com/benchmarks/street/index.html) · [Benchmark credits](https://movingai.com/benchmarks/grids.html) · [ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/) · [Benchmark paper](https://doi.org/10.1109/TCIAIG.2012.2197681)

