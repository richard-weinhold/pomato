import subprocess, sys, os
    
def instantiate_julia(package_path):

    args = ["julia", "_installation/julia_install_from_git.jl"] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=str(package_path)) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())

def instantiate_julia_dev(package_path, redundancyremoval_path, marketmodel_path):

    args = ["julia", "_installation/julia_install_dev.jl", redundancyremoval_path, marketmodel_path] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=str(package_path)) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())

def update_julia(package_path):

    args = ["julia", "_installation/julia_update.jl"] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=str(package_path)) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())

def add_gurobi(package_path):
    """Add Gurobi to Julia environment"""
    args = ["julia", "_installation/add_gurobi.jl"] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=str(package_path)) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())

def add_mosek(package_path):
    """Add Mosek to Julia environment"""
    args = ["julia", "_installation/add_mosek.jl"] 
    with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, cwd=str(package_path)) as program:
        for line in program.stdout:
            print(line.decode(errors="ignore").strip())

