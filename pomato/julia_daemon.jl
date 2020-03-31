
using JSON
# using RedundancyRemoval
# using MarketModel

function write_daemon_file(file)
    global daemon_file
    while true
        try
            io = open(daemon_file, "w")
            JSON.print(io, file, 2)
            close(io)
            break
        catch e
            @info("Failed to write to file $(daemon_file))")
        end
        sleep(1)
    end
end

function read_daemon_file()
    global daemon_file
    while true
        try
            io = open(daemon_file, "r")
            file = JSON.parse(io)
            close(io)
            return file
        catch e
            @info("Failed to read from file $(daemon_file))")
        end
        sleep(1)
    end
    
end

global model_type = ARGS[1]
# global model_type = "redundancy_removal"
global wdir = pwd()
global daemon_file = wdir*"/data_temp/julia_files/daemon_"*model_type*".json"
@info("reading from file $(daemon_file)")
file = read_daemon_file()

if model_type == "redundancy_removal"
    using RedundancyRemoval
elseif model_type == "market_model"
    using MarketModel
else
    throw(ArgumentError("No valid argument given"))
end

function run_redundancy_removal(file_suffix)
    if Threads.nthreads() >= 2
        @info("Run case $(file_suffix) on $(Threads.nthreads()) threads")
        RedundancyRemoval.run_redundancy_removal_parallel(file_suffix, filter_only=true)
    else
        @info("Run case $(file_suffix) single threaded")
        RedundancyRemoval.run_redundancy_removal(file_suffix)
    end
end

function run_market_model(wdir, data_dir, redispatch)
    if redispatch
        @info("Run market model with redispatch")
        MarketModel.run_market_model_redispatch(wdir, data_dir)
    else
        @info("Run market model")
        MarketModel.run_market_model(wdir, data_dir)
    end
    @info("Done with market model.")
end

@info("Done with Initialization. Starting daemon process $(file["type"])!")
while true
    file = read_daemon_file()
    if file["break"]
        @info("EXIT")
        break
    end
    sleep(1)
    if file["run"]
        file["run"] = false
        file["processing"] = true
        write_daemon_file(file)
        @info("Starting with $(file["type"])")
        if file["type"] == "redundancy_removal"
            file_suffix = file["file_suffix"]
            run_redundancy_removal(file_suffix)
        end
        if file["type"] == "market_model"
            # wdir = file["wdir"]
            global wdir
            redispatch = file["redispatch"]
            data_dir = file["data_dir"]
            run_market_model(wdir, data_dir, redispatch)
        end
        file["processing"] = false
        write_daemon_file(file)
    end
    # sleep(1)
    # if file["processing"]
    #     file["processing"] = false
    #     write_daemon_file(file)
    # end
    sleep(1)
    if !file["ready"]
        file["ready"] = true
        write_daemon_file(file)
    end
    # println("sleepy")
    sleep(1)
end
