############
#Read in the demand file
def prepare_demand_data(trips_file):
    file = open(trips_file)
    demands = Dict{(Int64,Int64), Float64}()
    s = 0
    for line in eachline(file)
        if contains(line, "Origin")
            s = int(split(line)[2])
        else
            pairs = split(line, ";")
            for pair in pairs
                if !contains(pair, "\n")
                    pair_vals = split(pair, ":")
                    t, demand = int(pair_vals[1]), float(pair_vals[2])
                    demands[(s,t)] = demand 
                end
            end
        end
    end                
    close(file)

    outfile = open("../temp_files/demands_Anaheim.json", "w")

    JSON.print(outfile, demands)

    close(outfile)