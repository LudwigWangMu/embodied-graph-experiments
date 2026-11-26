"""Rebuild maps, fit graph and dense selectors, evaluate policies, plot figures and run checks."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for name in ['make_tasks','evaluate','learn','comparators','navigation','handedness','report','verify']:
    subprocess.run([sys.executable,str(root/'code'/(name+'.py'))],cwd=root,check=True)

