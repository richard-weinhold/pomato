using Pkg
cd(@__DIR__)

Pkg.activate("pomato")
Pkg.add("Mosek")
Pkg.add("MosekTools")

