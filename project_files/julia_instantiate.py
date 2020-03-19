
import subprocess
from pathlib import Path

args = ["julia", "project_files/julia_instantiate.jl"]
with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                 	  stderr=subprocess.STDOUT, cwd=Path.cwd()) as programm:
    for line in programm.stdout:
    	print(line.decode(errors="ignore").strip())
    	