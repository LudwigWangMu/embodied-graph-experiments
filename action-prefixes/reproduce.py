"""Regenerate the archived numerical analyses and vector figures."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for name in ["experiment","physical_replay","closed_loop","reactive_control","render_alignment","calibration_budget","window_figure"]:
    subprocess.run([sys.executable,str(root/'code'/(name+'.py'))],cwd=root,check=True)

