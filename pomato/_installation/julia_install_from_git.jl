using Pkg

# runs from the pomato base folder
if isdir("_installation/pomato")
	rm("_installation/pomato", recursive=true)
end
Pkg.activate("_installation/pomato")
Pkg.add(PackageSpec(url="https://github.com/richard-weinhold/MarketModel.git", rev="master")) 
Pkg.add(PackageSpec(url="https://github.com/richard-weinhold/RedundancyRemoval.git", rev="master")) 
Pkg.add("JSON")
Pkg.add("Clp")

if "GUROBI_HOME" in keys(ENV)
	print("Adding Gurobi")
	Pkg.add("Gurobi")
end

Pkg.instantiate()
print("Precompiling...")
using JSON
using MarketModel
using RedundancyRemoval
