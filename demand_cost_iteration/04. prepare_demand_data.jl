function prepare_demand_data(out_dir, files_ID, time_instances, week_day_list, month_w)
    ############
    #Read in the demand file
    for instance in time_instances
        for day in week_day_list
            file = open( out_dir * "data_traffic_assignment_uni-class/" * files_ID * "_trips_" * month_w * "_" * string(day) * "_" * instance * ".txt")
            demands = Dict()
            s = 0
            cnt = 0
            for line in eachline(file)
                if contains(line, "Origin")
                    s = parse(Int, split(line)[2])

                elseif contains(line, " : ")
                    pairs = split(line, ";")

                    for pair in pairs
                        if contains(pair, ":")
                            pair_vals = split(pair, ":")
                            t = parse(Int, pair_vals[1])
                            demand = parse(Float64, pair_vals[2])
                            demands[(s,t)] = demand 
                        end
                    end
                end
            end                    
            
            close(file)
            
            outfile = open(out_dir * "/OD_demands/demands_" * month_w * "_" * string(day) * "_" * instance * ".json", "w")
            
            JSON.print(outfile, demands)
            
            close(outfile)
        end
        
    end
end


using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "");
@pyimport matplotlib.pyplot as plt
@pyimport numpy as np
@pyimport json
@pyimport os
@pyimport pickle
@pyimport pandas as pd
@pyimport collections

@pyimport parameters_julia
@pyimport utils_julia

out_dir = parameters_julia.out_dir;
files_ID = parameters_julia.files_ID;
month_w = parameters_julia.month_w;
year = parameters_julia.year;
time_instances = parameters_julia.instances_ID;
deg_grid = parameters_julia.deg_grid;
c_grid = parameters_julia.c_grid;
lamb_grid = parameters_julia.lamb_grid;
week_day_list = parameters_julia.week_day_list;

prepare_demand_data(out_dir, files_ID, time_instances, week_day_list, month_w)