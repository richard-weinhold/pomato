import subprocess, sys, os
from pathlib import Path
    
def instantiate_julia(gurobi=True, redundancyremoval_branch="master", marketmodel_branch="master"):
    cwd = str(Path(__file__).parent)
    args = ["julia", "julia_install_from_git.jl", redundancyremoval_branch, marketmodel_branch] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=cwd) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())
    if gurobi:
        add_gurobi()
        
def instantiate_julia_dev(redundancyremoval_path, marketmodel_path, gurobi=True):
    cwd = str(Path(__file__).parent)
    args = ["julia", "julia_install_dev.jl", redundancyremoval_path, marketmodel_path] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=cwd) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())

    if gurobi:
        add_gurobi()

def update_julia():
    cwd = str(Path(__file__).parent)
    args = ["julia", "julia_update.jl"] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=cwd) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())

def add_gurobi():
    """Add Gurobi to Julia environment"""
    cwd = str(Path(__file__).parent)
    args = ["julia", "add_gurobi.jl"] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=cwd) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())

def add_mosek():
    """Add Mosek to Julia environment"""
    cwd = str(Path(__file__).parent)
    args = ["julia", "add_mosek.jl"] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=cwd) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())

