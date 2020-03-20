using Pkg

Pkg.activate("pomato/_installation/pomato")

# Pkg.develop("pomato/RedundancyRemoval")
Pkg.develop(PackageSpec(path="pomato/RedundancyRemoval"))
Pkg.develop(PackageSpec(path="pomato/MarketModel"))
Pkg.add("JSON")
Pkg.instantiate()

print("precompiling")
using JSON
using MarketModel
using RedundancyRemoval
