
using CSV, DataFrames, Dates, Logging
using LinearAlgebra
using JuMP, GLPK
const MOI = JuMP.MathOptInterface

global wdir = "C:/Users/riw/tubcloud/Uni/Market_Tool/pomato/"
global suppress_csv_write = Dict("backup" => true,
								 "final" => false)

global_logger(ConsoleLogger(stdout, Logging.Debug))

function is_redundant(model, constraint::Vector{Float64}, rhs::Float64)
	temp = @constraint(model, constraint' * model[:x] <= rhs + 1)
	@objective(model, Max, constraint' * model[:x])
	JuMP.optimize!(model)
	JuMP.delete(model, temp)
	if JuMP.objective_value(model) > rhs
		result = []
		for i in model[:x]
			push!(result, JuMP.value(i))   
		end
		return true, result, model
	else 
		return false, [], model
	end
end

function build_model(n::Int, A::Array{Float64}, b::Vector{Float64})
	model = Model(with_optimizer(GLPK.Optimizer))
	@variable(model, x[i=1:n])
	@constraint(model, A * x .<= b)	
	return model
end

function add_constraint(model, constraint::Vector{Float64}, rhs::Float64)
	@constraint(model, sum(constraint .* model[:x]) <= rhs)
	return model
end

function RayShoot(A::Array{Float64}, b::Vector{Float64}, m::Vector{Int},
	              z::Vector{Float64}, r::Vector{Float64})
	stepsize = 1
	i = 0
	counter = 0
	while true
		point = z + r*i
		temp = A[m,:]*point - b[m]
		# If 1 constraint is hit, return it
		if size(findall(x-> x>0, temp), 1) == 1
			@debug ("Constraint hit: $(m[findall(x->x>0, temp)]) With i = $i")
			return m[findfirst(x->x>0, temp)]
		# If more than one constraint is hit:
		# 1) Check for breaking condition:
		#	 If True return either the first of the constraints hit
		#	 or all (currently only the first is returnd, since the others
		#	 will be checked at a lated stage since they will remain in m)
		#
		# 2) 1) Go one step back, to where no constraint was hit
		#    2) Continue forward with 1/of stepsize
		#	 3) Increase counter, since this procedure is limited by
		#	    17 decimal places for increment i 
		elseif size(findall(x-> x>0, temp))[1] > 1
			# Check breaking condition 
			if counter > 12
				@debug("Counter > 10, returning first of the constraints hit!")
				@debug("Constraints hit: $(m[findall(x->x>0, temp)]) With i = $i")
				return m[findfirst(x->x>0, temp)]
			else
				counter += 1
				# @debug("Going back and reducing stepsize. counter = ", counter)
				i = i - exp10(-stepsize)
				stepsize += 1
			end
		# No constraint is hit, incease increment i by stepsize
		else
			i = i + exp10(-stepsize)
		end
	end
end

function main(A::Array{Float64}, b::Vector{Float64}, 
			  m::Vector{Int}, I::Vector{Int}, z::Vector{Float64})
	@debug("Starting Algorithm with I of size: $(length(I))")
	@debug("and with m of size: $(length(m))")
	# Set-up 
	# Make counter to print out progress and make backups
	# every number of steps
	steps = 100
	to_check = length(m)
	stepsize = round(to_check/steps)
	save_points = [Int(x) for x in stepsize:stepsize:to_check]
	# Make sure only 100 save_points are available, remove the first ones
	if length(save_points) > 100
		save_points = save_points[(length(save_points) + 1 - 100):end]
	end
	### Build base model
	model = build_model(size(A, 2), A[I,:], b[I])
	# Start Algorithm
	while true
		k = m[1]
		@debug("checking constraint k = $k")
		# Check redundancy of constraint k
		# alpha, x_opt = @time is_redundant(model, A[k,:], b[k])
		alpha, x_opt = is_redundant(model, A[k,:], b[k])
		if alpha
			# If true, rayshoot and add constraint j to the model
			j = RayShoot(A, b, m, z, x_opt-z)
			@debug("k = $k and j = $j")
			model = add_constraint(model, A[j,:], b[j,])
			m = setdiff(m, j)
			I = union(I, j)
		else
			# if not, remove constraint from m
			m = setdiff(m, k)
		end
		# print progress and make backups of I, to allow for restart incase of crash :(
		if length(m) in save_points
			percentage = string(Int(100 - 100/steps*findfirst(x -> x==length(m), save_points)))
			@info("########## ------> "*percentage*"% done!")
			save_to_file(I, "backups/cbco_01_backup_"*percentage, suppress_csv_write["backup"])
		end
		if length(m) == 0
			break
		end
	end
	return I
end

function save_to_file(Indices, filename::String, suppress_write::Bool=false)
	# I_result.-1: Indices start at 0 (in python.... or any other decent programming language)
	if !suppress_write
		@debug("Writing File ", filename, " .... ")
		CSV.write(wdir*"/data_temp/julia_files/cbco_data/"*filename*".csv", 
		  	 	  DataFrame(constraints = Indices.-1))
	else
		@debug("No File Written bc of debug parameter in Line 8")
	end
end

function read_data(file_suffix::String)
	# Read Data From CSV Files
	@debug("Reading A, b Matrices...")
	# run_test = "test_full"
	# file_suffix = "test_pre"
	A_data = CSV.read(wdir*"/data_temp/julia_files/cbco_data/A_"*file_suffix*".csv", delim=',', header=false) 
	b_data = CSV.read(wdir*"/data_temp/julia_files/cbco_data/b_"*file_suffix*".csv", delim=',', header=false, types=Dict(1=>Float64) ) 

	A =  hcat([A_data[i] for i in 1:size(A_data, 2)]...)
	b = b_data[1]
	return A, b
end

function run(file_suffix::String)
	
	A, b = read_data(file_suffix)

	@info("Preprocessing...")
	@info("Removing duplicate rows...")
	# Remove douplicates
	condition_unique = .!nonunique(DataFrame(hcat(A,b)))
	@info("Removing all zero rows...")
	# Remove cb = co rows
	condition_zero = vcat([!all(A[i, :] .== 0) for i in 1:length(b)])

	I = Array{Int, 1}()
	m = collect(range(1, length(b)))
	m = m[condition_unique .& condition_zero]
	@info("Removed $(length(b) - length(m)) rows in preprocessing!")

	z = zeros(size(A, 2))

	I_result = @time main(A, b, m, I, z)

	@info("Number of non-redundant constraints: $(length(I_result))" )
	save_to_file(I_result, "cbco_01_"*file_suffix*"_"*Dates.format(now(), "ddmm_HHMM"), suppress_csv_write["final"])
end

function run_with_I(file_suffix::String, I_file::String, start_index::Int=1)
	A, b = read_data(file_suffix)
	# cbco_01_backup_91
	I_data = CSV.read(wdir*"/data_temp/julia_files/cbco_data/"*I_file*".csv", 
					  delim=',', types=Dict(1=>Int)) 
	I = I_data[1].+1
	@info("Preprocessing...")
	@info("Removing duplicate rows...")
	# Remove douplicates
	condition_unique = .!nonunique(DataFrame(hcat(A,b)))
	@info("Removing all zero rows...")
	# Remove cb = co rows
	condition_zero = vcat([!all(A[i, :] .== 0) for i in 1:length(b)])

	m = collect(range(1, length(b)))
	m = m[condition_unique .& condition_zero]
	m = filter(x -> x >= start_index, m)

	@info("Removed $(length(b) - length(m)) rows in preprocessing!")
	m = setdiff(m, I)

	z = zeros(size(A, 2))

	I_result = @time main(A, b, m, I, z)

	@info("Number of non-redundant constraints: $(length(I_result))")
	save_to_file(I_result, "cbco_01_I_"*file_suffix*"_"*Dates.format(now(), "ddmm_HHMM"), suppress_csv_write["final"])
end

### Make it rum from repl
if length(ARGS) > 0
	file_suffix = ARGS[1]
	global_logger(ConsoleLogger(stdout, Logging.Info))

    run_with_I(file_suffix, "I_"*file_suffix)
    # run(file_suffix)
end

#### 624
######################## Code Cemetery #########################
# inj_constraints = Diagonal(ones(size(A_data, 2)))
# upper_inj = ones(size(A_data, 2))*1e4
# A = vcat(inj_constraints, -inj_constraints, A)
# b = vcat(upper_inj, upper_inj, b) 
# tmp = I_result[I_result.>(size(A_data, 2)*2)].-(size(A_data, 2)*2)

# I = [2]
# m = collect(range(1, length(b)))
# z = zeros(2)
# alpha, x_opt = is_redundant(A, b, I, 1)
# j = RayShoot2(A, b, m, z, x_opt-z)


# Minimum BSP
# A = [
# 	 1 1;
# 	 1 0;
# 	 0 1;
# 	 0 1;
# 	 1 1;
# 	 ]
# b = [2.2 1 1 2 1.4]'