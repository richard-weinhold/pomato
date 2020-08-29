using Pkg

cd(@__DIR__)

# runs from the pomato base folder
if isdir("pomato")
	rm("pomato", recursive=true)
end
Pkg.activate("pomato")
Pkg.add(PackageSpec(url="https://github.com/richard-weinhold/MarketModel.git", rev="master")) 
Pkg.add(PackageSpec(url="https://github.com/richard-weinhold/RedundancyRemoval.git", rev="master")) 
Pkg.add("JSON")
Pkg.add("Clp")

Pkg.instantiate()
print("Precompiling...")
using JSON
using MarketModel
using RedundancyRemoval
