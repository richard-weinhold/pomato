using JSON 

WDIR = "C:\\Users\\riw\\tubCloud\\Uni\\Market_Tool\\pomato\\julia-files"
println("test")

model_type = "d2cf"
if in(model_type,  ["cbco_nodal", "cbco_zonal", "nodal"])
    println("Loading CBCOs")
    cbco = JSON.parsefile(WDIR*"/data/cbco.json")
elseif in(model_type,  ["d2cf"])
	cbco = JSON.parsefile(WDIR*"/data/cbco.json")
else
    cbco = Dict()
end


