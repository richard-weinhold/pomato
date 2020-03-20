
using JSON
# using RedundancyRemoval
# using MarketModel

function write_deamon_file(file)
    global deamon_file
    io = open(deamon_file, "w")
    JSON.print(io, file, 2)
    close(io)
end

function read_deamon_file()
    global deamon_file
    io = open(deamon_file, "r")
    file = JSON.parse(io)
    close(io)
    return file
end

global model_type = ARGS[1]
# global model_type = "redundancy_removal"
global wdir = pwd()
global deamon_file = wdir*"/data_temp/julia_files/deamon_"*model_type*".json"
@info("reading from file $(deamon_file))")
file = read_deamon_file()

if model_type == "redundancy_removal"
    using RedundancyRemoval
elseif model_type == "market_model"
    using MarketModel
else
    throw(ArgumentError("No valid argument given"))
end

function run_redundancy_removal(file_suffix)
    @info("Run case $(file_suffix) on $(Threads.nthreads()) threads")
    println("Here")
    RedundancyRemoval.run_redundancy_removal_parallel(file_suffix, filter_only=true)
end

function run_market_model(wdir, data_dir, redispatch)
    if redispatch
        @info("Run market model with redispatch")
        MarketModel.run_market_model_redispatch(wdir, data_dir)
    else
        @info("Run market model")
        MarketModel.run_market_model(wdir, data_dir)
    end
end

@info("Done with Initialization. Starting deamon process $(file["type"])!")
while true
    file = read_deamon_file()
    if file["break"]
        @info("EXIT")
        break
    end
    if file["run"]
        file["run"] = false
        file["processing"] = true
        write_deamon_file(file)
        @info("Starting with $(file["type"])")
        if file["type"] == "redundancy_removal"
            file_suffix = file["file_suffix"]
            run_redundancy_removal(file_suffix)
        end
        if file["type"] == "market_model"
            wdir = file["wdir"]
            redispatch = file["redispatch"]
            data_dir = file["data_dir"]
            run_market_model(wdir, data_dir, redispatch)
        end
        file["processing"] = false
        write_deamon_file(file)
    end
    if file["processing"]
        file["processing"] = false
        write_deamon_file(file)
    end
    if !file["ready"]
        file["ready"] = true
        write_deamon_file(file)
    end
    # println("sleepy")
    sleep(1)
end
