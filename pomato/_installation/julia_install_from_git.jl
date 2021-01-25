using Pkg

cd(@__DIR__)

redundancyremoval_branch = ARGS[1]
marketmodel_branch = ARGS[2]

# runs from the pomato base folder
if isdir("pomato")
	rm("pomato", recursive=true)
end
Pkg.activate("pomato")
Pkg.add(PackageSpec(url="https://github.com/richard-weinhold/MarketModel.git", rev=marketmodel_branch)) 
Pkg.add(PackageSpec(url="https://github.com/richard-weinhold/RedundancyRemoval.git", rev=redundancyremoval_branch)) 
Pkg.add("JSON")
Pkg.add("Clp")

Pkg.instantiate()
print("Precompiling...")
using JSON
using MarketModel
using RedundancyRemoval
