using JuMP
using Clp
using Ipopt
using JSON
using ECOS
using CSV

function remove_recundant_constraints(constraints, A, b, N, base_constraints)
	println("Removing redundant constraints...")
	# X = model[:X]
	redundant = []
	println("Number of CBCO Constraints in Model before removing redundant constraints: ",
			length(constraints))
	# Create Temp Model without constraint k
	tmp_model = Model(solver=IpoptSolver(print_level=0))
	@variable(tmp_model, X[N])
	@constraint(tmp_model, X[N] .<= 1e6)
	@constraint(tmp_model, X[N] .>= -1e6)
	@constraint(tmp_model, sum(X[n] for n in N) >= 0)

	# @constraint(tmp_model, sum(X[n] for n in N) == 0)

	for j in constraints
	# for j in setdiff(constraints, k)
		@constraint(tmp_model, sum(A[j,n]*X[n] for n in N)  <= b[j])
	end
	for k in constraints
		# Test constraints k to see if it still holds
    	@objective(tmp_model, Max, sum(A[k,n]*X[n] for n in N))
    	solve(tmp_model)
	    # if if b > obj k is redundant
	    if (b[k] - 1) > getobjectivevalue(tmp_model)
	        append!(redundant, k)
	    end
	end

	println("Number of redundant CBCO Constraints in Model: ", length(redundant))
	new_model = Model(solver=IpoptSolver(print_level=0))
	@variable(new_model, X[N])
	@constraint(new_model, X[N] .<= 1e6)
	@constraint(new_model, X[N] .>= -1e6)
	@constraint(new_model, sum(X[n] for n in N) >= 0)

	# @constraint(new_model, sum(X[n] for n in N) == 0)

	for j in setdiff(constraints, redundant)
		@constraint(new_model, sum(A[j,n]*X[n] for n in N)  <= b[j])
	end
	println("Number of CBCO Constraints in Model: ",
			MathProgBase.numlinconstr(new_model) - base_constraints)

	return new_model, setdiff(constraints, redundant)
end

### PROGRAMM START
WDIR = ARGS[1]
start_time = now()
# WDIR = "C:/Users/richi/tubCloud/Uni/Market_Tool/market_tool_julia/julia"
# WDIR = "C:/Users/riw/Documents/market_tool_git/julia"

println("Starting...")
# Read A and b for LP Ax<=b
println("Reading A, b...")
A = readcsv(WDIR*"/cbco_data/A.csv")
b = readcsv(WDIR*"/cbco_data/b.csv")
println("Reading A, b. Done!")
## This has to be of type INT, readcsv doesnt do that for some reason
## .. and indecies start at 1
init_constraints = CSV.read(WDIR*"/cbco_data/cbco_index.csv", datarow=1, types=[Int])
init_constraints = DataFrames.columns(init_constraints)[1].+1
# init_constraints = []

# b = b[:2000]
N = range(1, size(A)[2])
L = range(1, length(b))
# cbco = Model(solver=ClpSolver())
cbco = Model(solver=IpoptSolver(print_level=0))
# Variables
@variable(cbco, X[N])
@constraint(cbco, X[N] .<= 1e6)
@constraint(cbco, X[N] .>= -1e6)
@constraint(cbco, sum(X[n] for n in N) >= 0)

# @constraint(cbco, sum(X[n] for n in N) == 0)

base_constraints = MathProgBase.numlinconstr(cbco)
## Init Constraints from ConvexHull Presolve
for k in init_constraints
		@constraint(cbco, sum(A[k,n]*X[n] for n in N)  <= b[k])
end

println("Starting Algorithm with ", length(init_constraints), " initial constraints...")
constraints = init_constraints
# constraints = [x for x in range(1,144)]
add_constraints = 0
for i in setdiff(range(1, length(b)), init_constraints)
    @objective(cbco, Max, sum(A[i,n]*X[n] for n in N))
    solver_status = solve(cbco)
    if i%50 == 0
        println(i, "->", length(constraints))
    end
    if solver_status == :Infeasible
    	for n in N
    		setvalue(X[n], 0)
    	# solver_status = solve(cbco)
    	end
    end
    if ((b[i] + 1) < getobjectivevalue(cbco)) & !(solver_status == :Infeasible) # potentiall AND NOT instead of OR
        @constraint(cbco, sum(A[i,n]*X[n] for n in N)  <= b[i])
        append!(constraints,i)
        add_constraints += 1
    end
    if add_constraints > 100
    	cbco, constraints = remove_recundant_constraints(constraints, A, b, N, base_constraints)
    	X = cbco[:X]
    	add_constraints = 0
    end
end

cbco, constraints = remove_recundant_constraints(constraints, A, b, N, base_constraints)

writedlm(WDIR*"/cbco_data/cbco.csv", constraints.-1, ", ")

end_time = now()
println("Final Constraints: ", length(constraints))
println("Runtime: ", Dates.canonicalize(
					 Dates.CompoundPeriod(
					 Dates.Period(end_time - start_time))))
