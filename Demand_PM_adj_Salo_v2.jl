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
month = parameters_julia.month;
year = parameters_julia.year;
instances_1 = parameters_julia.instances_ID;
deg_grid = parameters_julia.deg_grid;
c_grid = parameters_julia.c_grid;
lamb_grid = parameters_julia.lamb_grid;
week_day_Apr_list = parameters_julia.week_day_list;




@pyimport GLS_julia
@pyimport Compute_Jacobian
@pyimport get_observed_flow

numNodes = Compute_Jacobian.numNodes;
numLinks = Compute_Jacobian.numLinks;
numODpairs = Compute_Jacobian.numODpairs;
numZones = Compute_Jacobian.numZones
od_pairs = Compute_Jacobian.od_pairs;
link_list_js = Compute_Jacobian.link_list_js;
link_length_list = Compute_Jacobian.link_length_list;





function read_demand_file(file_path, n_nodes);
    file = open(file_path)
    demands = Dict()
    n = n_nodes  # number of nodes
    for i = 1:n
        demands[(i,i)] = 0.0
    end
    for line in eachline(file)
        OD_demand = split(line, ",")
        key, value = (parse(Int, OD_demand[1]), parse(Int, OD_demand[2])), parse(Float64, split(OD_demand[3], "\n")[1])
        demands[key] = value
    end
    close(file)
    return demands
end


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



function load_ta_network_(out_dir, files_ID, month_w, day,  instan)
    toll_factor = 0
    distance_factor = 0
    network_data_file = files_ID * "_net_" * month_w * "_" * string(day) * '_' * instan * ".txt"
    trip_table_file = files_ID * "_trips_" * month_w * "_" * string(day) * '_' * instan * ".txt"
    best_objective = 0

    network_name = files_ID

    network_data_file =  out_dir  * "data_traffic_assignment_uni-class/" * network_data_file
    trip_table_file =   out_dir  * "data_traffic_assignment_uni-class/" * trip_table_file

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


type Arc
    initNode::Int 
    termNode::Int 
    capacity::Float64
    freeflowtime::Float64
    flow::Float64
end

Arc(initNode::Int, termNode::Int, capacity::Float64, freeflowtime::Float64) = 
    Arc(initNode, termNode, capacity, freeflowtime, 0.);


using Graphs

function create_graph(start_node, en_node)
    @assert Base.length(start_node)==Base.length(en_node)

    no_node = max(maximum(start_node), maximum(en_node))
    no_arc = Base.length(start_node)

    graph = simple_inclist(no_node)
    for i=1:no_arc 
        add_edge!(graph, start_node[i], en_node[i])
    end
    return graph
end

function get_vector(state, origin, destination, link_dic)
    current = destination
    parent = -1
    x = zeros(Int, maximum(link_dic))

    while parent != origin
        parent = state.parents[current]
        link_idx = link_dic[parent,current]
        if link_idx != 0
            x[link_idx] = 1
        end
        current = parent
    end
    return x
end

function BPR(flowVec, fcoeffs, free_flow_time, capacity)
    bpr = similar(flowVec)
    for a = 1:length(bpr)
        bpr[a] = free_flow_time[a] * sum([fcoeffs[i] * (flowVec[a]/capacity[a])^(i-1) for i = 1:length(fcoeffs)])
    end
    return bpr
end

function all_or_nothing(graph, link_dic, travel_time, demands, start_node, numZones)
    state = []
    path = []
    x = zeros(size(start_node))

    for r=1:numZones
        # for each origin node r, find shortest paths to all destination nodes
        state = dijkstra_shortest_paths(graph, travel_time, r)

        for s=1:numZones
            # for each destination node s, find the shortest-path vector
            # load travel demand
            x = x + demands[(r,s)] * get_vector(state, r, s, link_dic)
        end
    end

    return x
end


function iniDemand(trip_file, numZones, flag=0)
    file = open(trip_file)
    demands = Dict{}()
    for s=1:numZones
        for t=1:numZones
            demands[(s,t)] = 0
        end
    end    
    s = 0
    for line in eachline(file)
        if contains(line, "Origin")
            s = Int(parse(Float64,(split(line)[2])))
        elseif contains(line, ";")
            pairs = split(line, ";")
            for pair in pairs[1:end-1]
                if !contains(pair, "\n")
                    pair_vals = split(pair, ":")
                    t, demand = Int(parse(Float64,pair_vals[1])), parse(Float64,pair_vals[2])
                    demands[(s,t)] = demand 
                end
            end
        end
    end                        
    close(file)
    return demands
end


function tapMSA(graph, link_dic, demands, fcoeffs, free_flow_time, capacity, start_node, en_node, numZones, numIter=500, tol=1e-6)
    # Finding a starting feasible solution
    travel_time = BPR(zeros(numLinks), fcoeffs, free_flow_time, capacity)
    xl = all_or_nothing(graph, link_dic, travel_time, demands, start_node, numZones)

    l = 1

    while l < numIter
        l += 1

        xl_old = xl

        # Finding yl
        travel_time = BPR(xl, fcoeffs, free_flow_time, capacity )

        yl = all_or_nothing(graph, link_dic, travel_time, demands, start_node, numZones)

        # assert(yl != xl)

        xl = xl + (yl - xl)/l

        xl_new = xl

        relative_gap = norm(xl_new - xl_old, 1) / norm(xl_new, 1)

        if relative_gap < tol
            break
        end

    end

    tapFlows = Dict()

    for i = 1:length(start_node)
        key = (start_node[i], en_node[i])
        tapFlows[key] = xl[i]
    end

    tapFlowVect = xl

    return tapFlows, tapFlowVect
end

function extract_demandDict(out_dir, files_ID, month_w ,day, instance1, numZones)
    demandsDict = Dict()
    # get ground trueth demands, indexed by 0
    demandsDict[0] = iniDemand(out_dir * "data_traffic_assignment_uni-class/" * files_ID * "_trips_" * month_w * "_" * string(day) * "_" * instance1 * ".txt", numZones)
    # get initial demands, indexed by 1
    demandsDict[1] = iniDemand(out_dir * "data_traffic_assignment_uni-class/" * files_ID * "_trips_" * month_w * "_" * string(day) * "_" * instance1 * ".txt", numZones, 1)

    return demandsDict
end

function demandsDicToVec(demandsDic, odPairLabel_)
    demandsVec = zeros(length(odPairLabel_))
    for i = 1:length(demandsVec)
        demandsVec[i] = demandsDic[(odPairLabel_["$i"][1], odPairLabel_["$i"][2])]
    end
    return demandsVec
end

function demandsVecToDic(demandsVec, odPairLabel_)
    demandsDic = Dict{}()
    for i = 1:numNodes
        demandsDic[(i, i)] = 0
    end
    for i = 1:length(demandsVec)
        demandsDic[(odPairLabel_["$i"][1], odPairLabel_["$i"][2])] = demandsVec[i]
    end
    return demandsDic
end

function arcData(arc_file)
    arcs = Dict()
    file = open(arc_file)
    inHeader=true
    for line in eachline(file)
        if inHeader
            inHeader = !contains(line, "Init node")
            continue
        end
        vals = split(line, )
        arcs[(parse(Int, vals[1]), parse(Int, vals[2]))] = Arc(parse(Int, vals[1]), parse(Int, vals[2]), parse(Float64, vals[3]), parse(Float64, vals[5]))
    end
    close(file) 
    return arcs
end


# add flow data to arcs
function observFlow(arc_file, tapFlowDic)
    arcs = arcData(arc_file)
    ix = 0 
    for key in keys(arcs)
        arcs[key].flow = tapFlowDic[key]
    end
    return arcs
end


function objF(graph, link_dic, gamma1, gamma2, demandsVec, demandsVec0, tapFlowVecDict, fcoeffs, free_flow_time, capacity, start_node, en_node, numZones, odPairLabel_)
    demandsDic = demandsVecToDic(demandsVec, odPairLabel_)
    tapFlowVec = tapMSA(graph, link_dic, demandsDic, fcoeffs, free_flow_time, capacity, start_node, en_node, numZones)[2]
    return gamma1 * sum([(demandsVec[i] - demandsVec0[i])^2 for i = 1:length(demandsVec)]) + gamma2 * sum([(tapFlowVec[a] - tapFlowVecDict[a])^2 for a = 1:length(tapFlowVec)])
end     

function tapFlowVecToLinkCostDict(tapFlowVec, fcoeffsInvVI, free_flow_time, capacity)
    linkCostVec = BPR(tapFlowVec, fcoeffsInvVI, free_flow_time, capacity)
    temp_dict = Dict{}()
    for i in 1:length(linkCostVec)
        temp_dict["$(i-1)"] = linkCostVec[i]
    end
    return temp_dict
end

# compute the gradient
function gradient_(gamma1, gamma2, demandsVec, demandsVec0, tapFlowVec, observFlowVec, jacob, numODpairs, numLinks)
    gradi = zeros(numODpairs)
    for i = 1:numODpairs
        gradi[i] = 2 * gamma1 * (demandsVec[i] - demandsVec0[i]) + 2 * gamma2 * sum([(tapFlowVec[j] - observFlowVec[j]) * jacob[i, j] for j = 1:numLinks])
    end
    return gradi
end
    
function descDirec(gamma1, gamma2, demandsVec, demandsVec0, tapFlowVec, observFlowVec, jacob, numODpairs, numLinks)
    gradi = gradient_(gamma1, gamma2, demandsVec, demandsVec0, tapFlowVec, observFlowVec, jacob, numODpairs, numLinks)
    h = similar(gradi)
    for i = 1:length(gradi)
        h[i] = -1 * gradi[i]
    end
    return h
end


# compute a search direction
function searchDirec(demandsVec, descDirect, epsilon_1)
    h = descDirect
    h_ = similar(h)
    for i = 1:length(h)
            if (demandsVec[i] > epsilon_1) || (demandsVec[i] <= epsilon_1 && h[i] > 0)
            h_[i] = h[i]
        else
            h_[i] = 0
        end
    end
    return h_
end

# line search
function thetaMax(demandsVec, searchDirect)
    h_ = searchDirect
    thetaList = Float64[]
    for i = 1:length(h_)
        if h_[i] < 0
            push!(thetaList, - demandsVec[i]/h_[i])
        end
    end
    theta_max = minimum(thetaList)
    return theta_max
end



function armijo(gamma1, gamma2, objFunOld, demandsVecOld, demandsVec0, tapFlowVecDict, fcoeffs, searchDirec, thetaMax, Theta, N, graph, link_dic, free_flow_time, capacity, start_node, en_node, numZones , odPairLabel_)
    demandsVecList = Array{Float64}[]
    objFunList = Float64[]
    push!(demandsVecList, demandsVecOld)
    push!(objFunList, objFunOld)
    for n = 0:N
        demandsVecNew = similar(demandsVecOld)
        for i = 1:length(demandsVecOld)
            demandsVecNew[i] = demandsVecOld[i] + (thetaMax/(Theta^n)) * searchDirec[i] 
        end
        
        objFun_New = objF(graph, link_dic, gamma1, gamma2, demandsVecNew, demandsVec0, tapFlowVecDict, fcoeffs, free_flow_time, capacity, start_node, en_node, numZones , odPairLabel_)
        push!(demandsVecList, demandsVecNew)
        push!(objFunList, objFun_New)
    end
    idx = indmin(objFunList)
    objFunNew = objFunList[idx]
    assert(objFunNew <= objFunOld)
    return demandsVecList[idx], objFunNew
end

function BPRSocial(flowVec, fcoeffs, free_flow_time, capacity)
    bpr = similar(flowVec)
    # refer to [Page 50; Patriksson 1994, 2015]
    for a = 1:length(bpr)
        bpr[a] = free_flow_time[a] * sum([fcoeffs[i] * (flowVec[a]/capacity[a])^(i-1) for i = 1:length(fcoeffs)])
        + free_flow_time[a] * sum([fcoeffs[i] * (i-1) * (flowVec[a]/capacity[a])^(i-1) for i = 2:length(fcoeffs)])
    end
    return bpr
end


function tapMSASocial(demands, fcoeffs, graph, link_dic, start_node, en_node, free_flow_time, capacity, numLinks, numZones, numIter=1000, tol=1e-6)
    # Finding a starting feasible solution
    travel_time = BPRSocial(zeros(numLinks), fcoeffs, free_flow_time, capacity)
    xl = all_or_nothing(graph, link_dic, travel_time, demands, start_node, numZones)

    l = 1

    while l < numIter
        l += 1
        xl_old = xl

        # Finding yl
        travel_time = BPRSocial(xl, fcoeffs, free_flow_time, capacity)
        yl = all_or_nothing(graph, link_dic, travel_time, demands, start_node, numZones)
        
        # assert(yl != xl)
        xl = xl + (yl - xl)/l
        xl_new = xl

        relative_gap = norm(xl_new - xl_old, 1) / norm(xl_new, 1)
        if relative_gap < tol
            break
        end

    end
    

    tapFlows = Dict()

    for i = 1:length(start_node)
        key = (start_node[i], en_node[i])
        tapFlows[key] = xl[i]
    end

    tapFlowVect = xl

    return tapFlows, tapFlowVect
end


function socialObj(linkFlowVec, free_flow_time, polyDeg, fcoeffs, capacity, numLinks)
    objVal = sum([sum([free_flow_time[a] * fcoeffs[i] * linkFlowVec[a]^i / capacity[a]^(i-1) for i=1:polyDeg]) 
        for a = 1:numLinks])
    return objVal
end


using JuMP, Ipopt
function socialOpt(out_dir, files_ID, instance1, demandsVec, polyDeg, free_flow_time, fcoeffs, capacity)
  
    #load OD pair-route incidence
    odPairRoute = readstring(out_dir * "od_pair_route_incidence_" * instance1 *  files_ID * ".json");
    odPairRoute = JSON.parse(odPairRoute);

    for route in keys(odPairRoute)
        if odPairRoute[route]>0
            odPairRoute[route] = 1
        end
    end
    
    #load link labels
    linkLabel = readstring(out_dir * "link_label_dict.json");
    linkLabel = JSON.parse(linkLabel);
    
    #load link-route incidence
    linkRoute = readstring(out_dir * "link_route_incidence_" * instance1 *  files_ID * ".json");
    linkRoute = JSON.parse(linkRoute);
    
    
    m = Model(solver=IpoptSolver());
    numLinks = length(linkLabel)
    numRoute = length(linkRoute)
    #numRoute = 400

    numOD = length(demandsVec)

    @variable(m, linkFlow[1:numLinks])
    @variable(m, pathFlow[1:numRoute])

    pathFlowSum = Dict()

    for i=1:numOD
        pathFlowSum[i] = 0
        for j=1:numRoute
            if "$(i)-$(j)" in keys(odPairRoute)
                pathFlowSum[i] += pathFlow[j]
            end
        end
        @constraint(m, pathFlowSum[i] == demandsVec[i])
    end

    pathFlowLinkSum = Dict()



    for a=1:numLinks
        pathFlowLinkSum[a] = 0
        for j=1:numRoute
            if "$(a)-$(j)" in keys(linkRoute)
                pathFlowLinkSum[a] += pathFlow[j];
            end
        end
        @constraint(m, pathFlowLinkSum[a] == linkFlow[a]);
    end



    for j=1:numRoute
        @constraint(m, pathFlow[j] >= 0);
    end


    for a=1:numLinks
        @constraint(m, pathFlowLinkSum[a] <= capacity[a]);
    end

    #@expression(m, f, sum{free_flow_time[a]*linkFlow[a] + .03*free_flow_time[a]*((linkFlow[a])^5)/((capacity[a])^4), a = 1:numLinks} )


#   @NLexpression(m, f, sum{ free_flow_time[a] * fcoeffs[i]  *linkFlow[a]^i / capacity[a]^(i-1) , i = 1:polyDeg , a = 1:numLinks }) ;

    @NLexpression(m, f, sum{free_flow_time[a] * fcoeffs[1] * linkFlow[a] +
            free_flow_time[a] * fcoeffs[2] * linkFlow[a]^2 / capacity[a] +
            free_flow_time[a] * fcoeffs[3] * linkFlow[a]^3 / capacity[a]^2 +
            free_flow_time[a] * fcoeffs[4] * linkFlow[a]^4 / capacity[a]^3 +
            free_flow_time[a] * fcoeffs[5] * linkFlow[a]^5 / capacity[a]^4 +
            free_flow_time[a] * fcoeffs[6] * linkFlow[a]^6 / capacity[a]^5 +
            free_flow_time[a] * fcoeffs[7] * linkFlow[a]^7 / capacity[a]^6 +#, a = 1:numLinks})
            free_flow_time[a] * fcoeffs[8] * linkFlow[a]^8 / capacity[a]^7, a = 1:numLinks})

    @NLobjective(m, Min, f);
    #print(m) 

    solve(m);
    
    return getobjectivevalue(m)
end



objFunDict = Dict()

function adjustingODdemands(instance, day, cnt, out_dir, month_w, month, year, files_ID, ta_data, link_list,numNodes, gamma1, gamma2 )
    ## Load initialization 
    flow_observ, y, link_vector = get_observed_flow.get_observed_flow(out_dir, files_ID, instance, month_w,  month, year, week_day_Apr_list);

    # demands from GLS
    g_0 = read_demand_file(out_dir * "OD_demands/OD_demand_matrix_" * month_w * "_" * string(day) * "_weekday_" * instance * files_ID * ".txt", numNodes)


    # cost function parameters
    coeffs_dict_Apr_PM_ = readstring(out_dir * "coeffs_dict_" * month_w *  "_" * instance * ".json");
    coeffs_dict_Apr_PM_ = JSON.parse(coeffs_dict_Apr_PM_);

    # cross-validation best key selection
    best_key = readstring(out_dir * "cross_validation_best_key/cross_validation_best_key_" * month_w * "_" * string(day) * "_" * instance * ".json");
    best_key = JSON.parse(best_key);
    best_key = "(8, 0.5, 1.0, 2)"
    fcoeffs = coeffs_dict_Apr_PM_[best_key];
    #fcoeffs = [1,0,0,0,.5,0,0,0]
    #fcoeffs = [1.0,-0.00302509, 0.0577279, -0.195632, 0.620696, -0.905963, 0.936143, -0.469483, 0.108584];
    polyDeg = length(fcoeffs);

    # load network
    ta_data = load_ta_network_(out_dir, files_ID, month_w, day,  instance);

    # Load OdPairLabel
    odPairLabel_ = readstring(out_dir * "od_pair_label_dict__refined.json");
    odPairLabel_ = JSON.parse(odPairLabel_);

    # Renaming variables
    numNodes = maximum(map(pair->pair[1], keys(g_0)));
    start_node = ta_data.start_node;
    end_node = ta_data.end_node;
    capacity = ta_data.capacity;
    free_flow_time = ta_data.free_flow_time;
    number_of_zones = ta_data.number_of_zones;
    numLinks = size(start_node)[1];

    numODpairs = numNodes * (numNodes - 1);
    graph = create_graph(start_node, end_node);
    link_list = sparse(start_node, end_node, 1:numLinks);

    # Run MSA and adjust demand vector
    tapFlows = Dict();
    tapFlowVect = Dict();
    tapFlowDicDict = Dict();
    tapFlowVecDict = Dict();
    linkCostDicDict = Dict();
    jacobiSpiessDict = Dict();
    jacobDict = Dict();
    descDirecDict = Dict();
    searchDirecDict = Dict();
    thetaMaxDict = Dict();
    demandsVecDict = Dict();
    demandsDiffDict = Dict();
    norObjFunDict = Dict();
    tapSocialFlowDicDict = Dict();
    tapSocialFlowVecDict = Dict();
    tapSocialFlowDicDict_ = Dict();
    tapSocialFlowVecDict_ = Dict();
    user_sol_dict = Dict();
    social_sol_dict = Dict();
    PoA_dict = Dict();
    
    #demandsVecDict[0] = demandsDicToVec(g_0, odPairLabel_);
    #demandsDiffDict[1] = norm(demandsDicToVec(demandsDict[1], odPairLabel_) - demandsDicToVec(g_0, odPairLabel_))/
    #                     norm(demandsDicToVec(g_0, odPairLabel_));

    if isdir(out_dir * "demandsDict") == false
            mkdir(out_dir * "demandsDict");
    end

    #obj_dict[day] = demandsDictFixed(demandsDict, flow_observ,link_vector, graph, ta_data, link_dic, day, gamma1, gamma2, 
    #    out_dir, files_ID, month_w, instance, key_, free_flow_time, capacity, start_node, en_node, numZones, cnt)


    xl = flow_observ[:, cnt]
    tapFlows[0] = Dict()
    for i = 1:length(start_node)
        key = link_vector[i]
        tapFlows[0][key] = xl[i]
    end

    tapFlowVecDict[0] = [];
    for i = 1:length(start_node)
        key = link_vector[i];
        append!(tapFlowVecDict[0], tapFlows[0][key])
    end

    network_data_file = files_ID * "_net_" * month_w * "_full_" * instance * ".txt";
    arcsDict = observFlow(out_dir  * "data_traffic_assignment_uni-class/" * network_data_file, tapFlows[0]);


    demandsVecDict[0] = demandsDicToVec(g_0, odPairLabel_);
    #demandsVecDict[0] = ones(length(demandsVecDict[0]))    
    demandsVecDict[1] = demandsDicToVec(g_0, odPairLabel_);
    #demandsVecDict[1] = ones(length(demandsVecDict[1]))    


    objFunDict[1] = objF(graph, link_list, gamma1, gamma2, demandsVecDict[0], 
        demandsVecDict[0], tapFlowVecDict[0], fcoeffs, free_flow_time, capacity, start_node, end_node, numZones, odPairLabel_);

    demandsDic = demandsVecToDic(demandsVecDict[0], odPairLabel_)

    # get initial flow vector (corresponding to initial demands)
    tapFlows[1], tapFlowVecDict[1] = tapMSA(graph, link_list, g_0, fcoeffs, free_flow_time, capacity, start_node, end_node, number_of_zones);

    # Computation of a Descent direction 
    linkCostDicDict[1] = tapFlowVecToLinkCostDict(tapFlowVecDict[1], fcoeffs, free_flow_time, capacity);
    #linkCostDicDict[1]["0"], link_length_list[1]
    jacobiSpiessDict[1] = Compute_Jacobian.jacobianSpiess(numNodes, numLinks, numODpairs, od_pairs, link_list_js, [linkCostDicDict[1]["$(i)"] for i=0:numLinks-1]);

    demandsDict = extract_demandDict(out_dir, files_ID, month_w ,day, instance, numZones);

    # maximum number of iterations
    N = 100;

    # Armijo rule parameters
    rho = 2;
    M = 10;

    # search direction parameter
    epsilon_1 = 0;

    # stop criterion parameter
    epsilon_2 = 1e-20;
    obj = 9999999999999

    las = 0

    for l = 1:N
        jacobDict[l] = jacobiSpiessDict[l]

        descDirecDict[l] = descDirec(gamma1, gamma2, demandsVecDict[l], demandsVecDict[0],  tapFlowVecDict[l],
                tapFlowVecDict[0], jacobDict[l], numODpairs, numLinks);

        demandsVecDict[l] = demandsDicToVec(demandsDict[l], odPairLabel_);

        searchDirecDict[l] = searchDirec(demandsVecDict[l], descDirecDict[l], epsilon_1);

        thetaMaxDict[l] = thetaMax(demandsVecDict[l], searchDirecDict[l]);

        demandsVecDict[l+1] = similar(demandsVecDict[0]);

        demandsVecDict[l+1], objFunDict[l+1] = armijo(gamma1, gamma2, objFunDict[l], demandsVecDict[l], demandsVecDict[0], tapFlowVecDict[0], fcoeffs, 
                searchDirecDict[l], thetaMaxDict[l], rho, M, graph, link_list, free_flow_time, capacity, start_node, end_node, numZones, odPairLabel_);

        demandsDict[l+1] = demandsVecToDic(demandsVecDict[l+1], odPairLabel_);

        tapFlows[l+1], tapFlowVecDict[l+1] = tapMSA(graph, link_list, demandsDict[l+1], fcoeffs, free_flow_time, capacity, start_node, end_node, numZones);

        arcsDict[l+1] = observFlow(out_dir  * "data_traffic_assignment_uni-class/" * network_data_file, tapFlows[l+1]);

        linkCostDicDict[l+1] = tapFlowVecToLinkCostDict(tapFlowVecDict[l+1], fcoeffs, free_flow_time, capacity);

        jacobiSpiessDict[l+1] = Compute_Jacobian.jacobianSpiess(numNodes, numLinks, numODpairs, od_pairs,
                                                      link_list_js, [linkCostDicDict[l+1]["$(i)"] for i=0:numLinks-1]);

        demandsDiffDict[l+1] = norm(demandsVecDict[l+1] - demandsVecDict[0]) / norm(demandsVecDict[0]);

        obj =  objFunDict[1] - objFunDict[l]

        las = l
            # stopping criterion
        if (objFunDict[l] - objFunDict[l+1]) / objFunDict[1] < epsilon_2
            break
        end


        println("iteration $(l) finished...")

    end

    # normalize objective function value
    for a = 1:(length(objFunDict))
        norObjFunDict[a] = objFunDict[a] / objFunDict[1];
    end

    # write files
    outfile = open(out_dir * "demandsDict/demandsVecDict$(day)_" * month_w * "_" * instance * ".json", "w")
    JSON.print(outfile, demandsVecDict)
    close(outfile)

    outfile = open(out_dir * "demandsDict/demandsDict$(day)_" * month_w * "_" * instance * ".json", "w")
    JSON.print(outfile, demandsDict)
    close(outfile)

    outfile = open(out_dir * "demandsDict/tapFlowDicDict$(day)_" * month_w * "_" * instance * ".json", "w")
    JSON.print(outfile, tapFlows)
    close(outfile)

    outfile = open(out_dir * "demandsDict/tapFlowVecDict$(day)_" * month_w * "_" * instance * ".json", "w")
    JSON.print(outfile, tapFlowVecDict)
    close(outfile)

    outfile = open(out_dir * "demandsDict/jacobi$(day)_" * month_w * "_" * instance * ".json", "w")
    JSON.print(outfile, jacobiSpiessDict[las])
    close(outfile)

    demandsDict[length(demandsDict)-1]
    demandsDict_ = Dict{}()

    for key in keys(demandsDict[length(demandsDict)-1])
        demandsDict_[key] = demandsDict[length(demandsDict)-1][key]
    end

    outfile = open(out_dir * "demandsDict/demandsDictFixed$(day)_"* month_w * "_" * instance * ".json", "w")
    JSON.print(outfile, demandsDict_)
    close(outfile)
    
    # Calculating POA
    tapSocialFlowDicDict_[day], tapSocialFlowVecDict_[day] =  tapMSASocial(demandsDict[las], fcoeffs, graph, link_list, start_node, end_node, free_flow_time, capacity, numLinks, numZones);
    #tapSocialFlowDicDict[day] = socialOpt(out_dir, files_ID, instance, demandsVecDict[las], polyDeg, free_flow_time, fcoeffs, capacity)
    #print(tapSocialFlowDicDict[day])
    
    #user_sol_dict[day] = socialObj(flow_observ[:, cnt], free_flow_time, polyDeg, fcoeffs, capacity, numLinks) ;
    user_sol_dict[day] = socialObj(tapFlowVecDict[las], free_flow_time, polyDeg, fcoeffs, capacity, numLinks) ;
    tapSocialFlowDicDict[day] = socialObj(tapSocialFlowVecDict_[day], free_flow_time, polyDeg, fcoeffs, capacity, numLinks);
    
    PoA_dict[day] = user_sol_dict[day] / tapSocialFlowDicDict[day];
    
    println("-----------------------------------------")
    println("day $(day) finished...")
    println("PoA for day $(day) and instance " * instance * " is " * string(PoA_dict[day]))
    println("-----------------------------------------")
    
    return PoA_dict[day], user_sol_dict[day], obj
end






PoA_dict = Dict()
day = 3
cnt = 2
instance = "PM"
gamma1 = 0
gamma2 = 1
ta_data = load_ta_network_(out_dir, files_ID, month_w, day,  instance);
link_list = sparse(ta_data.start_node, ta_data.end_node, 1:numLinks);
instances_1 = ["PM"]
cnt = 0
for instance in instances_1
    PoA_dict = Dict()
    cong = Dict()
    obj = Dict()
    for day in week_day_Apr_list
        cnt += 1
        PoA_dict[day], cong[day], obj[day] = adjustingODdemands(instance, day, cnt, out_dir, month_w, month, year, files_ID, ta_data, link_list, numNodes, gamma1, gamma2 )
    end
    
    outfile =  open(out_dir * "cong_" * month_w * "_" * instance * ".json", "w")
    JSON.print(outfile, cong)
    close(outfile)

    outfile =  open(out_dir * "PoA_dict_" * month_w * "_" * instance * ".json", "w")
    JSON.print(outfile, PoA_dict)
    close(outfile)

    outfile =  open(out_dir * "obj_dict" * month_w * "_" * instance * ".json", "w")
    JSON.print(outfile, obj)
    close(outfile)  
end