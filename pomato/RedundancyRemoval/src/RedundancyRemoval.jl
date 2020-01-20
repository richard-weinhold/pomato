module RedundancyRemoval

using CSV, DataFrames, Dates, Logging
using LinearAlgebra
using JuMP, GLPK, Gurobi

include("redundancy_removal_functions.jl")

export run

function __init__()
	global wdir = pwd()
	global_logger(ConsoleLogger(stdout, Logging.Info))
	@info("No arguments passed or not running in repl, initializing in pwd()")
	@info("Initialized")
end

end # module
