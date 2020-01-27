
function is_redundant(model::JuMP.Model, constraint::Vector{Float64}, rhs::Float64)
	temp = @constraint(model, constraint' * model[:x] <= rhs + 1)
	@objective(model, Max, constraint' * model[:x])
	JuMP.optimize!(model)
	JuMP.delete(model, temp)
	@debug("Solition", JuMP.value.(model[:x]))
	if JuMP.objective_value(model) > rhs
		# result = JuMP.value.(model[:x])
		return true, JuMP.value.(model[:x]), model
	else
		return false, JuMP.value.(model[:x]), model
	end
end

function build_model(dim::Int, A::Array{Float64},
    				 b::Vector{Float64}, x_bounds::Vector{Float64})
	# model = Model(with_optimizer(GLPK.Optimizer, msg_lev=GLPK.OFF))
	model = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0, Method=0,
				  Presolve=0, PreDual=0, Aggregate=0))
	if size(x_bounds, 1) > 0
		@info("Building Model with bounds on x!")
		@variable(model, x[i=1:dim], lower_bound=-x_bounds[i], upper_bound=x_bounds[i])
	else
		@info("Building Model with x free!")
		@variable(model, x[i=1:dim])
	end

	@constraint(model, con[i=1:size(A, 1)], A[i,:]' * x <= b[i])
	@constraint(model, sum(x[i] for i in 1:dim) == 0)
	return model
end

function add_constraint(model::JuMP.Model, constraint::Vector{Float64}, rhs::Float64)
	@constraint(model, sum(constraint .* model[:x]) <= rhs)
	return model
end

function RayShoot(A::Array{Float64}, b::Vector{Float64}, m::Vector{Int},
	              z::Vector{Float64}, x_opt::Vector{Float64})
	### Set-up
	stepsize = 1
	i = 0
	counter = 0
	max_iterations = 10
	## Define Ray (vector from interior point z to x_op)
	r = x_opt - z
	## Subset m to only contain constraints violated by x_opt
	m_hat = m[findall(x-> x>0, A[m,:]*x_opt - b[m])]
	# m_hat = m
	if length(m_hat) == 0
		 @info("WARNING: m_hat empty, possibly numerical error")
		 @info("Moving further outside along the Ray")
		 m_hat = m[findall(x-> x>=0, A[m,:]*x_opt*1.01 - b[m])]
		 @info("m_hat $(m_hat), length = $(length(m_hat))!")
	end

	while true
		point = z + r*i
		temp = A[m_hat,:]*point - b[m_hat]
		# If 1 constraint is hit, return it
		constraints_hit = size(findall(x-> x>0, temp), 1)
		if constraints_hit == 1
			@debug("Constraint hit: $(m_hat[findall(x->x>0, temp)]) With i = $i, stepsize = $stepsize")
			return m_hat[findfirst(x->x>0, temp)]
		# If more than one constraint is hit:
		# 1) Check for breaking condition:
		#	 If True return either the first of the constraints hit
		# 2) 1) Go one step back, to where no constraint was hit
		#    2) Continue forward with 1/10 of stepsize
		#	 3) Increase counter, since this procedure is limited by
		#	    float number of decimal places for increment i
		elseif constraints_hit > 1
			# Check breaking condition
			if counter > max_iterations
				@debug("Counter > $(max_iterations), returning first of the constraints hit!")
				@debug("Constraints hit: $(m_hat[findall(x->x>0, temp)]) With i = $i")
				return m_hat[findfirst(x->x>0, temp)]
			else
				# Going back and reducing stepsize by 10th
				counter += 1
				i = i - exp10(-stepsize)
				stepsize += 1
			end
		# No constraint is hit, incease increment i by stepsize
		else
			i = i + exp10(-stepsize)
		end
	end
end

function main(A::Array{Float64}, b::Vector{Float64},  m::Vector{Int},
			  I::Vector{Int}, x_bounds::Vector{Float64}, z::Vector{Float64})
	@info("Starting Algorithm with I of size: $(length(I))")
	@info("and with m of size: $(length(m))")
	# Set-up
	# Make counter to print out progress every number of steps
	steps = 5
	to_check = length(m)
	stepsize = round(to_check/steps)
	save_points = [Int(x) for x in stepsize:stepsize:to_check]
	# Make sure only 100 save_points are available, remove the first ones
	if length(save_points) > steps
		save_points = save_points[(length(save_points) + 1 - steps):end]
	end
	### Build base model
	model = build_model(size(A, 2), A[I,:], b[I], x_bounds)
	# Start Algorithm
	while true
		k = m[1]
		@debug("checking constraint k = $k")
		# Check redundancy of constraint k
		alpha, x_opt = is_redundant(model, A[k,:], b[k])
		if alpha
			# If true, rayshoot and add constraint j to the model
			j = RayShoot(A, b, m, z, x_opt)
			@debug("k = $k and j = $j")
			model = add_constraint(model, A[j,:], b[j])
			m = setdiff(m, j)
			push!(I, j)
		else
			# if not, remove constraint from m
			m = setdiff(m, k)
		end
		# print progress at specified when size(m) hit certain threasholds
		if length(m) in save_points
			percentage = Int(100 - 100/steps*findfirst(x -> x==length(m), save_points))
			progress_bar = repeat("#", Int(round(percentage/5)))*repeat(" ", Int(round((100-percentage)/5)))
			timestamp = Dates.format(now(), "dd.mm - HH:MM:SS")
			report = "- Size of I $(length(I)) - Found Redundant $(to_check - length(I) - length(m)) - Remaining $(length(m)) - "
			thread = " on Thread: "*string(Threads.threadid())
			@info(progress_bar*string(percentage)*"%"*report*timestamp*thread)
		end

		### Conclude when m is empty
		if length(m) == 0
			break
		end
	end
	return I
end

function split_m(m::Vector{Int}, splits::Int)
	m_segments = []
	segment_length = Int(floor(length(m)/splits))
	for i in 1:splits - 1
		push!(m_segments, m[(i-1)*segment_length + 1:i*segment_length])
	end
	push!(m_segments, m[(splits-1)*segment_length + 1:end])
 	return m_segments
end

function parallel_filter(A::Array{Float64}, b::Vector{Float64},
						 m::Vector{Int}, x_bounds::Vector{Float64},
						 z::Vector{Float64}, splits::Int)

	m_segments = split_m(m, Int(splits))
	lock = SpinLock()
	indices = Array{Int, 1}()
	Threads.@threads for m_seg in m_segments
		idx = main(A, b, m_seg, Array{Int, 1}(), x_bounds, z)
		withlock(lock) do
			indices = union(indices, idx)
		end
		@info("Nonredundant ", length(idx), " from process ", Threads.threadid())
	end
	println("Length of m: ", length(indices))
	return indices
end

function solve_parallel!(model::JuMP.Model, constraint::Vector{Float64},
						rhs::Float64, k::Int)
	@objective(model, Max, constraint' * model[:x])
	JuMP.delete(model, model[:con][k])
	JuMP.optimize!(model)
	if JuMP.objective_value(model) < rhs
		return false
	else
		@constraint(model, sum(constraint .* model[:x]) <= rhs)
		return true
	end
end

function main_parallel(A::Array{Float64}, b::Vector{Float64},
					   m::Vector{Int}, I::Vector{Int}, x_bounds::Vector{Float64},
					   z::Vector{Float64})

	filtered_m = copy(m)
	filter_splits = Threads.nthreads()*2
	while true
		tmp_m = parallel_filter(A, b, filtered_m, x_bounds, z, Int(filter_splits))
		println("m ", length(filtered_m), " m new: ", length(tmp_m))
		if Int(filter_splits) == Threads.nthreads()/2
			println("breaking..")
			filtered_m = tmp_m
			break
		else
			filter_splits = Int(filter_splits/2)
		end
		filtered_m = tmp_m
	end
	lock = SpinLock()
	I = Array{Int, 1}()

	number_ranges = maximum(Int, [floor(Int, length(filtered_m)/1000),
								  Threads.nthreads()])

	ranges = split_m(filtered_m, number_ranges)
	Threads.@threads for r in ranges
		@info("LP Test with length ", length(r), " on proc id: ", Threads.threadid())
		kk = zeros(Bool, length(r))
		model_copy = build_model(size(A, 2), A[filtered_m,:], b[filtered_m], x_bounds)
		for k in 1:length(r)
			@inbounds kk[k] = solve_parallel!(model_copy, A[r[k], :], b[r[k]], findfirst(x -> x == r[k], filtered_m))
		end
		withlock(lock) do
			I = union(I, r[kk])
		end
	end
	return I
end

function save_to_file(Indices, filename::String)
	# I_result.-1: Indices start at 0 (in python.... or any other decent programming language)
	@info("Writing File "*filename*" .... ")
	CSV.write(wdir*"/data_temp/julia_files/cbco_data/"*filename*".csv",
	  	 	  DataFrame(constraints = Indices.-1))
	@info("done! ")
end

function read_data(file_suffix::String)
	# Read Data From CSV Files
	@info("Reading A, b, x_bounds and I Matrices...")
	I = Array{Int, 1}()
	A_data = CSV.read(wdir*"/data_temp/julia_files/cbco_data/A_"*file_suffix*".csv",
					  delim=',', header=false)
	b_data = CSV.read(wdir*"/data_temp/julia_files/cbco_data/b_"*file_suffix*".csv",
					  delim=',', header=false, types=Dict(1=>Float64))

	# Create Array A and Vector b from DataFrame
	A =  hcat([A_data[:, i] for i in 1:size(A_data, 2)]...)
	b = b_data[:,1]

	x_bounds = CSV.read(wdir*"/data_temp/julia_files/cbco_data/x_bounds_"*file_suffix*".csv",
					    delim=',', header=false, types=Dict(1=>Float64))
	# Read X Bounds or set as empty Vector
	x_bounds = size(x_bounds, 2) > 0 ? x_bounds[:,1] : Array{Float64, 1}()
	# x_bounds = Array{Float64, 1}()

	# I data contrains previsouly identified non-redundant indices
	I_data = CSV.read(wdir*"/data_temp/julia_files/cbco_data/I_"*file_suffix*".csv",
					  delim=',', types=Dict(1=>Int))
  	I = size(I_data, 2) > 0 ? I_data[:,1] : Array{Int, 1}()

	return A, b, x_bounds, I
end

function run(file_suffix::String)
	A, b, x_bounds, I = read_data(file_suffix)

	m = collect(1:length(b))
	@info("Preprocessing...")
	@info("Removing duplicate rows...")
	# Remove douplicates
	condition_unique = .!nonunique(DataFrame(hcat(A,b)))
	@info("Removing all zero rows...")
	# Remove cb = co rows
	condition_zero = vcat([!all(A[i, :] .== 0) for i in 1:length(b)])
	m = m[condition_unique .& condition_zero]
	@info("Removed $(length(b) - length(m)) rows in preprocessing!")
	# Interior point z = zero
	z = zeros(size(A, 2))
	I = union(I)
	m = setdiff(m, I)
	I_result = main(A, b, m, I, x_bounds, z)

	@info("Number of non-redundant constraints: $(length(I_result))" )
	save_to_file(I_result, "cbco_"*file_suffix*"_"*Dates.format(now(), "ddmm_HHMM"))
	@info("Everything Done!")
end

function run_parallel(file_suffix::String)
	A, b, x_bounds, I = read_data(file_suffix)
	m = collect(1:length(b))
	@info("Preprocessing...")
	@info("Removing duplicate rows...")
	# Remove douplicates
	# condition_unique = .!nonunique(DataFrame(hcat(A,b)))
	@info("Removing all zero rows...")
	# Remove cb = co rows
	# condition_zero = vcat([!all(A[i, :] .== 0) for i in 1:length(b)])
	# m = m[condition_unique .& condition_zero]
	@info("Removed $(length(b) - length(m)) rows in preprocessing!")
	# Interior point z = zero
	z = zeros(size(A, 2))
	I = union(I)
	m = setdiff(m, I)
	I_result = main_parallel(A, b, m, I, x_bounds, z)

	@info("Number of non-redundant constraints: $(length(I_result))" )
	save_to_file(I_result, "cbco_"*file_suffix*"_"*Dates.format(now(), "ddmm_HHMM"))
	@info("Everything Done!")
end
