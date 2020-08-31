using Pkg
cd(@__DIR__)

Pkg.activate("pomato")
if !("Gurobi" in keys(Pkg.installed()))
    if "GUROBI_HOME" in keys(ENV)
        Pkg.add("Gurobi")
    else
        print("GUROBI_HOME not on path! Gurobi not installed!")
    end
end
