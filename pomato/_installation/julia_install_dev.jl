using Pkg
cd(@__DIR__)

marketmodel_path = ARGS[1]
redundancyremoval_path = ARGS[2]
# runs from the pomato base folder
if isdir("pomato")
	rm("pomato", recursive=true)
end
Pkg.activate("pomato")
Pkg.develop(PackageSpec(url=marketmodel_path))
Pkg.develop(PackageSpec(url=redundancyremoval_path))

Pkg.add("JSON")
Pkg.add("Clp")
Pkg.add("ECOS")

Pkg.instantiate()
print("Precompiling...")
using JSON
using MarketModel
using RedundancyRemoval
