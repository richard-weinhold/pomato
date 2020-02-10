module RedundancyRemoval

using CSV, DataFrames, Dates, Logging, LoggingExtras
using LinearAlgebra
using JuMP, Gurobi
using Base.Threads, ThreadTools

include("redundancy_removal_functions.jl")

export run, run_parallel

function __init__()
	global wdir = pwd()
	# global global_logger(ConsoleLogger(stdout, Logging.Info))
	TeeLogger(FileLogger(wdir*"/logs/RedundancyRemoval.log", append=true),
	          ConsoleLogger(stdout, Logging.Info)) |> global_logger
	@info("No arguments passed or not running in repl, initializing in pwd()")
	@info("Initialized")
end

end # module
