module RedundancyRemoval

using CSV, DataFrames, Dates, Logging, LoggingExtras
using LinearAlgebra
using JuMP, Gurobi
using Base.Threads, ThreadTools

include("redundancy_removal_functions.jl")

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

	# @require Gurobi="2e9cd046-0924-5485-92f1-d5272153d98b" begin
	# 	println("Gurobi Init.")
	# 	function return_optimizer()
	# 		@info("Using Gurobi optimizer")
	# 		return with_optimizer(Gurobi.Optimizer, OutputFlag=0, Method=0,
	# 							  Presolve=0, PreDual=0, Aggregate=0)
	# 	end
	# end
	# println("Initialized")
end

export run_redundancy_removal_parallel, run_redundancy_removal

end # module
