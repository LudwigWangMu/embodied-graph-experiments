# Embodied graph experiments

Computational materials for four manuscripts on graph models and embodied decision making. Each study has its own entry point, dependency list, recorded outputs, and data-source instructions.

| Study | Start here | An independently checkable result |
| --- | --- | --- |
| How Far Does a Push Travel? Paired Action Responses in Contact-Graph World Models | [Contact response](contact-response/) | Recompute the error change from adding the graph correction in each of eight fork and packing comparisons. |
| Episode-Calibrated Action Prefixes from Robot–Object Graph Forecasts | [Action prefixes](action-prefixes/) | Aggregate the frozen-policy episode logs at 120 and 300 actions and reconcile violations with policy calls. |
| Reusing Scene-Graph Readouts Across Asynchronous Robot Updates | [Graph readouts](graph-readouts/) | Count stale releases and compare ordering rules against the recorded per-packet oracle. |
| Which Way Should a Robot Look? Decision-Relevant Symmetry Breaking on Navigation Graphs | [Active frame sensing](active-frame-sensing/) | Recover action risks and charged route costs from per-task and per-route records. |

## Verify the archived results

Download the repository at the manuscript's commit, or clone it and check out the release tag. From the repository root, run the following with Python 3.12. This path is offline and uses the standard library.

```sh
python verify.py
```

The verifier checks SHA-256 digests and recomputes selected manuscript results from rows of individual scenes, episodes, packets or routes. For one study, add its directory name, for example `python verify.py --study action-prefixes`.

## Rerun an experiment

Open the corresponding directory and follow its README. Use a separate environment per study. The CPU entry point in each directory is `python reproduce.py`. Action-prefix replay and graph-readout replay have a data-preparation step. The two frozen visual policies and the GPU board-energy measurement have separate hardware and commands.

Full runs write new tables and figures in the working copy. Use a second checkout to preserve the archived measurements. The original checksums refer to the distributed files. Use `python verify.py --results-only` to compare a rerun's detailed records to its summaries. Host latencies and board energy depend on the machine.

Third-party input data and model checkpoints are credited in each study directory. Numerical units, evaluation splits, seeds, grouping, and device details are recorded with the corresponding experiments. The versioned release accompanies the manuscripts for review.
