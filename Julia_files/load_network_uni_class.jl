# Sioux Falls network data
# http://www.bgu.ac.il/~bargera/tntp/

#Link travel time = free flow time * ( 1 + B * (flow/capacity)^Power ).
#Link generalized cost = Link travel time + toll_factor * toll + distance_factor * distance

# Traffic Assignment Data structure
type TA_Data
    network_name::String

    number_of_zones::Int64
    number_of_nodes::Int64
    first_thru_node::Int64
    number_of_links::Int64

    start_node::Array
    end_node::Array
    capacity::Array
    link_length::Array
    free_flow_time::Array
    B::Array
    power::Array
    speed_limit::Array
    toll::Array
    link_type::Array

    total_od_flow::Float64

    travel_demand::Array
    od_pairs::Array

    toll_factor::Float64
    distance_factor::Float64

    best_objective::Float64
end



function load_ta_network_(out_dir, files_ID, month_w, instan)
    toll_factor = 0
    distance_factor = 0
    network_data_file = files_ID * "_net_" * month_w * "_" * instan * ".txt"
    trip_table_file = files_ID * "_trips_" * month_w * "_" * instan * ".txt"
    best_objective = 0

    network_name = files_ID

    network_data_file =  out_dir * files_ID * "/data_traffic_assignment_uni-class/" * network_data_file
    trip_table_file =   out_dir * files_ID * "/data_traffic_assignment_uni-class/" * trip_table_file

    number_of_zones = 0
    number_of_links = 0
    number_of_nodes = 0
    first_thru_node = 0
    
    n = 0
    A = 0
    n = open(network_data_file, "r")
    A = readlines(n)
    for line in A
        #print( line)
        if contains(line, "<NUMBER OF ZONES>")
            number_of_zones = parse( Int, line[ search(line, '>')+1 : end] )
        elseif contains(line, "<NUMBER OF NODES>")
            number_of_nodes = parse( Int, line[ search(line, '>')+1 : end] )
        elseif contains(line, "<FIRST THRU NODE>")
            first_thru_node = parse( Int, line[ search(line, '>')+1 : end] )
        elseif contains(line, "<NUMBER OF LINKS>")
            number_of_links = parse( Int, line[ search(line, '>')+1 : end] )
        elseif contains(line, "<END OF METADATA>")
       # println(line)
        end
    end

    @assert number_of_links > 0
    start_node = Array{Int}(number_of_links)
    end_node = Array{Int}(number_of_links)
    capacity = zeros(number_of_links)
    link_length = zeros(number_of_links)
    free_flow_time = zeros(number_of_links)
    B = zeros(number_of_links)
    power = zeros(number_of_links)
    speed_limit = zeros(number_of_links)
    toll = zeros(number_of_links)
    link_type = Array{Int}(number_of_links)

    idx = 1

    apa = 0
    for line in A
        if contains(line, "~")
            continue
        end

        if contains(line, ";")
            line = strip(line, '\n')
            line = strip(line, ';')
            numbers = split(line)
            start_node[idx] = parse(Int, numbers[1])
            end_node[idx] = parse(Int, numbers[2])
            
            capacity[idx] = parse(Float64, numbers[3])
            link_length[idx] = parse(Float64, numbers[4])
            
            free_flow_time[idx] = parse(Float64, numbers[5])
            B[idx] = parse(Float64, numbers[6])
            power[idx] = parse(Float64, numbers[7])
            speed_limit[idx] = parse(Float64, numbers[8])
            toll[idx] = parse(Float64, numbers[9])
            link_type[idx] = parse(Float64, numbers[10])
            
            idx = idx + 1
        end
    end

    ##################################################
    # Trip Table
    ##################################################

    number_of_zones_trip = 0
    total_od_flow = 0
    
    f = open(trip_table_file, "r")
    fe = readlines(f)
    
    for line in fe
        #println(line)
        if contains(line, "<NUMBER OF ZONES>")
            number_of_zones_trip = parse( Int, line[ search(line, '>')+1 : end ] )
        elseif contains(line, "<TOTAL OD FLOW>")
            total_od_flow = parse( Float64, line[ search(line, '>')+1 : end ] )
        elseif contains(line, "<END OF METADATA>")
            break
        end
    end

    @assert number_of_zones_trip == number_of_zones # Check if number_of_zone is same in both txt files

    travel_demand = zeros(number_of_zones, number_of_zones)
    od_pairs = []
    origin = 0
    for line in fe
        if contains(line, "Origin")
            origin = parse( Int, split(line)[2] )
        elseif contains(line, ";")
            pairs = split(line, ";")
            for i=1:size(pairs)[1]
                if contains(pairs[i], ":")
                    pair = split(pairs[i], ":")
                    destination = parse( Int, strip(pair[1]) )
                    od_flow = parse( Float64, strip(pair[2]) )
                    travel_demand[origin, destination] = od_flow
                    push!(od_pairs, (origin, destination))
                   # println("origin=$origin, destination=$destination, flow=$od_flow")
                end
            end
        end
    end

    # Preparing data to return
    ta_data = TA_Data(
        network_name,
        number_of_zones,
        number_of_nodes,
        first_thru_node,
        number_of_links,
        start_node,
        end_node,
        capacity,
        link_length,
        free_flow_time,
        B,
        power,
        speed_limit,
        toll,
        link_type,
        total_od_flow,
        travel_demand,
        od_pairs,
        toll_factor,
        distance_factor,
        best_objective)

    return ta_data

end # end of load_network 




#=
function load_ta_network(network_name="Sioux Falls")

    toll_factor = 0
    distance_factor = 0

    if network_name == "Sioux Falls"
        network_data_file = "SiouxFalls_net.txt"
        trip_table_file = "SiouxFalls_trips.txt"
        best_objective = 0
    elseif network_name == "Sioux_simp"
        network_data_file = "SiouxFalls_net_simp.txt"
        trip_table_file = "SiouxFalls_trips_simp.txt"
        best_objective = 0
    elseif network_name == "Anaheim"
        network_data_file = "Anaheim_net.txt"
        trip_table_file = "Anaheim_trips.txt"
        best_objective = 0
    elseif network_name == "Tiergarten"
        network_data_file = "Tiergarten_net.txt"
        trip_table_file = "Tiergarten_trips.txt"
        best_objective = 0
    elseif network_name == "MA"
        network_data_file = "MA_net.txt"
        trip_table_file = "MA_trips.txt"
        best_objective = 0
    elseif network_name == "MA_journal_Apr_PM"
        network_data_file = "journal_net_Apr_PM.txt"
        trip_table_file = "journal_trips_Apr_PM.txt"
        best_objective = 0
    elseif network_name == "MA_journal_Apr_weekend"
        network_data_file = "journal_net_Apr_weekend.txt"
        trip_table_file = "journal_trips_Apr_weekend.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jan_AM"
        network_data_file = "East_Massachusetts_net_Jan_AM.txt"
        trip_table_file = "East_Massachusetts_trips_Jan_AM.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jan_MD"
        network_data_file = "East_Massachusetts_net_Jan_MD.txt"
        trip_table_file = "East_Massachusetts_trips_Jan_MD.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jan_PM"
        network_data_file = "East_Massachusetts_net_Jan_PM.txt"
        trip_table_file = "East_Massachusetts_trips_Jan_PM.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jan_NT"
        network_data_file = "East_Massachusetts_net_Jan_NT.txt"
        trip_table_file = "East_Massachusetts_trips_Jan_NT.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jan_weekend"
        network_data_file = "East_Massachusetts_net_Jan_weekend.txt"
        trip_table_file = "East_Massachusetts_trips_Jan_weekend.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Apr_AM"
        network_data_file = "East_Massachusetts_net_Apr_AM.txt"
        trip_table_file = "East_Massachusetts_trips_Apr_AM.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Apr_MD"
        network_data_file = "East_Massachusetts_net_Apr_MD.txt"
        trip_table_file = "East_Massachusetts_trips_Apr_MD.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Apr_PM"
        network_data_file = "East_Massachusetts_net_Apr_PM.txt"
        trip_table_file = "East_Massachusetts_trips_Apr_PM.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Apr_NT"
        network_data_file = "East_Massachusetts_net_Apr_NT.txt"
        trip_table_file = "East_Massachusetts_trips_Apr_NT.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Apr_weekend"
        network_data_file = "East_Massachusetts_net_Apr_weekend.txt"
        trip_table_file = "East_Massachusetts_trips_Apr_weekend.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jul_AM"
        network_data_file = "East_Massachusetts_net_Jul_AM.txt"
        trip_table_file = "East_Massachusetts_trips_Jul_AM.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jul_MD"
        network_data_file = "East_Massachusetts_net_Jul_MD.txt"
        trip_table_file = "East_Massachusetts_trips_Jul_MD.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jul_PM"
        network_data_file = "East_Massachusetts_net_Jul_PM.txt"
        trip_table_file = "East_Massachusetts_trips_Jul_PM.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jul_NT"
        network_data_file = "East_Massachusetts_net_Jul_NT.txt"
        trip_table_file = "East_Massachusetts_trips_Jul_NT.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Jul_weekend"
        network_data_file = "East_Massachusetts_net_Jul_weekend.txt"
        trip_table_file = "East_Massachusetts_trips_Jul_weekend.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Oct_AM"
        network_data_file = "East_Massachusetts_net_Oct_AM.txt"
        trip_table_file = "East_Massachusetts_trips_Oct_AM.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Oct_MD"
        network_data_file = "East_Massachusetts_net_Oct_MD.txt"
        trip_table_file = "East_Massachusetts_trips_Oct_MD.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Oct_PM"
        network_data_file = "East_Massachusetts_net_Oct_PM.txt"
        trip_table_file = "East_Massachusetts_trips_Oct_PM.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Oct_NT"
        network_data_file = "East_Massachusetts_net_Oct_NT.txt"
        trip_table_file = "East_Massachusetts_trips_Oct_NT.txt"
        best_objective = 0
    elseif network_name == "East_Massachusetts_Oct_weekend"
        network_data_file = "East_Massachusetts_net_Oct_weekend.txt"
        trip_table_file = "East_Massachusetts_trips_Oct_weekend.txt"
        best_objective = 0
    elseif network_name == "Sioux Falls simplified"
        network_data_file = "SiouxFalls_net_simplified.txt"
        trip_table_file = "SiouxFalls_trips_simplified.txt"
        best_objective = 0
    end


    #root_dir = "C:/Users/Salomon Wollenstein/Documents/GitHub/InverseVIsTraffic/"
    #network_data_file = joinpath(root_dir, joinpath("data_traffic_assignment_uni_class", network_data_file))
    #trip_table_file = joinpath(root_dir, joinpath("data_traffic_assignment_uni_class", trip_table_file))




    ##################################################
    # Network Data
    ##################################################


    #network_data_file = files_ID * "_net_" * month_w * "_" * instan * ".txt"

    number_of_zones = 0
    number_of_links = 0
    number_of_nodes = 0
    first_thru_node = 0

    n = open(network_data_file, "r")

    while (line=readline(n)) != ""
        if contains(line, "<NUMBER OF ZONES>")
            number_of_zones = parse( Int, line[ search(line, '>')+1 : end-1 ] )
        elseif contains(line, "<NUMBER OF NODES>")
            number_of_nodes = parse( Int, line[ search(line, '>')+1 : end-1 ] )
        elseif contains(line, "<FIRST THRU NODE>")
            first_thru_node = parse( Int, line[ search(line, '>')+1 : end-1 ] )
        elseif contains(line, "<NUMBER OF LINKS>")
            number_of_links = parse( Int, line[ search(line, '>')+1 : end-1 ] )
        elseif contains(line, "<END OF METADATA>")
            break
        end
    end

    @assert number_of_links > 0

    start_node = Array(Int, number_of_links)
    end_node = Array(Int, number_of_links)
    capacity = zeros(number_of_links)
    link_length = zeros(number_of_links)
    free_flow_time = zeros(number_of_links)
    B = zeros(number_of_links)
    power = zeros(number_of_links)
    speed_limit = zeros(number_of_links)
    toll = zeros(number_of_links)
    link_type = Array(Int, number_of_links)

    idx = 1
    while (line=readline(n)) != ""
        if contains(line, "~")
            continue
        end

        if contains(line, ";")
            line = strip(line, '\n')
            line = strip(line, ';')

            numbers = split(line)

            start_node[idx] = parse(Int, numbers[1])
            end_node[idx] = parse(Int, numbers[2])
            capacity[idx] = parse(Float64, numbers[3])
            link_length[idx] = parse(Float64, numbers[4])
            free_flow_time[idx] = parse(Float64, numbers[5])
            B[idx] = parse(Float64, numbers[6])
            power[idx] = parse(Float64, numbers[7])
            speed_limit[idx] = parse(Float64, numbers[8])
            toll[idx] = parse(Float64, numbers[9])
            link_type[idx] = parse(Float64, numbers[10])

            idx = idx + 1
        end
    end

    ##################################################
    # Trip Table
    ##################################################

    number_of_zones_trip = 0
    total_od_flow = 0

    f = open(trip_table_file, "r")

    while (line=readline(f)) != ""
        if contains(line, "<NUMBER OF ZONES>")
            number_of_zones_trip = parse( Int, line[ search(line, '>')+1 : end-1 ] )
        elseif contains(line, "<TOTAL OD FLOW>")
            total_od_flow = parse( Float64, line[ search(line, '>')+1 : end-1 ] )
        elseif contains(line, "<END OF METADATA>")
            break
        end
    end

    @assert number_of_zones_trip == number_of_zones # Check if number_of_zone is same in both txt files

    travel_demand = zeros(number_of_zones, number_of_zones)
    od_pairs = []
    while (line=readline(f)) != ""
        if contains(line, "Origin")
            origin = parse( Int, split(line)[2] )
        elseif contains(line, ";")
            pairs = split(line, ";")
            for i=1:size(pairs)[1]
                if contains(pairs[i], ":")
                    pair = split(pairs[i], ":")
                    destination = parse( Int, strip(pair[1]) )
                    od_flow = parse( Float64, strip(pair[2]) )
                    travel_demand[origin, destination] = od_flow
                    push!(od_pairs, (origin, destination))
                    # println("origin=$origin, destination=$destination, flow=$od_flow")
                end
            end
        end
    end

    # Preparing data to return
    ta_data = TA_Data(
        network_name,
        number_of_zones,
        number_of_nodes,
        first_thru_node,
        number_of_links,
        start_node,
        end_node,
        capacity,
        link_length,
        free_flow_time,
        B,
        power,
        speed_limit,
        toll,
        link_type,
        total_od_flow,
        travel_demand,
        od_pairs,
        toll_factor,
        distance_factor,
        best_objective)

    return ta_data

end # end of load_network function
=#
