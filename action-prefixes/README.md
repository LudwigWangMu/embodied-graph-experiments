# Episode-Calibrated Action Prefixes from Robot–Object Graph Forecasts

The replay experiments use Push-T scene coordinates and MCube physical coordinates in millimetres. The two visual-policy studies log separate simulator runs. Their calibration and test sets differ from the demonstration-replay sets.

## CPU replay and closed-loop retrieval

Use Python 3.12 in a new environment. From this directory run

```sh
python -m pip install -r requirements.txt
python prepare_inputs.py
python reproduce.py
```

Input preparation downloads the original Push-T archive and the selected MCube repeatability records. It checks the Push-T archive digest and all 206 raw MCube member digests before reconstructing the four input arrays. Existing copies can be reused with `--pusht PATH_TO/pusht.zip`, `--mcube-cache PATH_TO/raw_members`, or `--mcube-archive PATH_TO/rect1_json.zip`. The root verifier reads the supplied result tables directly without these downloads. [Push-T release](https://diffusion-policy.cs.columbia.edu/) · [MIT MCube data and acquisition protocol](https://mcube.mit.edu/push-dataset/)

The physical inputs are aligned causally at 50 Hz. The nine body-frame probes are reconstructed from the measured object pose. The calibration, fit, scale and test assignments appear in `data/split.csv` and `data/physical_split.csv`. For primary replay results see `data/coverage.csv` and `data/physical_coverage.csv`. Closed-loop and reactive outcomes appear in `data/closed_loop_episodes.csv` and `data/reactive_episodes.csv`. The retrieval-policy calibration uses resets 31000–31047 and the test uses 32000–32059.

## Frozen visual controllers

Use a separate Windows x64 / Python 3.12 environment with an NVIDIA driver compatible with the recorded CUDA 12.8 PyTorch build. Complete CPU input preparation first. The files in `data/` can be shared between environments.

```sh
python -m pip install --no-deps -r requirements_vla_windows.txt
python code/frozen_vla.py
python code/check_vla.py
python code/visual_diffusion.py
python code/diffusion_horizon.py
python code/plot_diffusion_pairs.py
```

The model loaders select fixed upstream revisions. For offline use, pass `--policy PATH` and, for SmolVLA, `--backbone PATH` to the corresponding evaluation script. The GPU access, model downloads and training-library installation are unnecessary for the archived-result verifier. [SmolVLA action checkpoint](https://huggingface.co/smy1729/smolvlapusht/tree/c9525f2afb45503a8319e353aa33b597efe48359) · [Backbone configuration](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/4cde9d208e19519c36917aa170fffc7e381aa0ad) · [Visual diffusion checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653)

Each mode has 24 test resets. Check `data/frozen_vla_episodes.csv` against `data/frozen_vla_summary.json` and `data/visual_diffusion_episodes.csv` against `data/visual_diffusion_summary.json`. The 300-action fixed-eight extension uses the same 24 seeds in `data/visual_diffusion_horizon.csv`. The root verifier reconstructs mean peak overlap, successes, violations and policy calls. The driving code and output files carry the original experiment seeds and budget.
