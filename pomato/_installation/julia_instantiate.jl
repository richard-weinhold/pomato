using Pkg

if isdir("pomato/_installation/pomato")
	rm("pomato/_installation/pomato", recursive=true)
end
Pkg.activate("pomato/_installation/pomato")
Pkg.develop(PackageSpec(path="pomato/RedundancyRemoval"))
Pkg.develop(PackageSpec(path="pomato/MarketModel"))
Pkg.add("JSON")
Pkg.instantiate()

print("Precompiling...")

using JSON
using MarketModel
using RedundancyRemoval
