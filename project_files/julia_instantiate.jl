using Pkg

Pkg.activate("project_files\\pomato")
Pkg.instantiate()

using JSON
using MarketModel
using RedundancyRemoval
