"""Regenerate the archived numerical analyses and vector figures."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for name in ["experiment","learned_baseline","edge_audit","distance_profile","action_selection","axial_baseline","hybrid_response","residual_figure","branch_transfer","real_reset_pairs","matched_geometry","geometry_panels"]:
    subprocess.run([sys.executable,str(root/'code'/(name+'.py'))],cwd=root,check=True)

