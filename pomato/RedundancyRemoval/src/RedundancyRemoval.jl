module RedundancyRemoval

using CSV, DataFrames, Dates, Logging, LoggingExtras
using LinearAlgebra
using JuMP, Gurobi
using Base.Threads, ThreadTools

include("redundancy_removal_functions.jl")

export run_redundancy_removal_parallel, run_redundancy_removal

function __init__()
	global wdir = pwd()
	# global_logger(ConsoleLogger(stdout, Logging.Info))
	if isfile(wdir*"/logs/RedundancyRemoval.log")
		TeeLogger(MinLevelLogger(FileLogger(wdir*"/logs/RedundancyRemoval.log", append=true), Logging.Info),
		          ConsoleLogger(stdout, Logging.Info)) |> global_logger
		println("Logfile Found, logging to console and logfile.")
	else
		TeeLogger(ConsoleLogger(stdout, Logging.Info)) |> global_logger
		println("No logfile Found, logging only to console.")
	end
	println("Initialized")
end
end # module


