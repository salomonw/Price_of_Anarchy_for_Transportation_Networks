# read in arc data

type Arc
    initNode::Int 
    termNode::Int 
    capacity::Float64
    freeflowtime::Float64
    flow::Float64
end

Arc(initNode::Int, termNode::Int, capacity::Float64, freeflowtime::Float64) = 
    Arc(initNode, termNode, capacity, freeflowtime, 0.)

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

# read in initial demand data
srand(1617)
function iniDemand(trip_file, flag=0)
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

function demandsDicToVec(demandsDic)
    demandsVec = zeros(length(odPairLabel_))
    for i = 1:length(demandsVec)
        demandsVec[i] = demandsDic[(odPairLabel_["$i"][1], odPairLabel_["$i"][2])]
    end
    return demandsVec
end

function demandsVecToDic(demandsVec)
    demandsDic = Dict{}()
    for i = 1:numNodes
        demandsDic[(i, i)] = 0
    end
    for i = 1:length(demandsVec)
        demandsDic[(odPairLabel_["$i"][1], odPairLabel_["$i"][2])] = demandsVec[i]
    end
    return demandsDic
end

# obtain important parameters of the network

include("Julia_files/load_network_uni_class.jl")

function paraNetwork(out_dir, files_ID, month_w, instance1)
    ta_data = load_ta_network_(out_dir, files_ID, month_w, instance1)
    numNodes = maximum(map(pair->pair[1], keys(demandsDict[0])))
    start_node = ta_data.start_node
    capacity = ta_data.capacity
    free_flow_time = ta_data.free_flow_time
    numLinks = size(start_node)[1]
    numODpairs = numNodes * (numNodes - 1)
    return numNodes, numLinks, numODpairs, capacity, free_flow_time, ta_data
end

function tapFlowVecToLinkCostDict(tapFlowVec, fcoeffsInvVI)
    linkCostVec = BPR(tapFlowVec, fcoeffsInvVI)
    temp_dict = Dict{}()
    for i in 1:length(linkCostVec)
        temp_dict["$(i-1)"] = linkCostVec[i]
    end
    return temp_dict
end

# calculate od Pair - route choice probability matrix P

function odPairRouteProb(tau, route_cost_vec, OD_pair_route_dict, numODpairs, numRoutes)
    P = zeros((numODpairs, numRoutes))
    for i in 1:numODpairs
        for r in OD_pair_route_dict["$(i-1)"]
    #         println(r)
    #         break
            # P[i, r] = 1
            P[i, r+1] = exp(- tau * route_cost_vec[r+1]) / sum([exp(- tau * route_cost_vec[j+1]) 
                for j in OD_pair_route_dict["$(i-1)"]])
        end
    end

    # maximum(P[9,:])

    P_dict = Dict{}()
    for i in 1:numODpairs
        for j in 1:numRoutes
            key = "$(i)-$(j)"
            if (P[i, j] > 1e-15)
                P_dict[key] = P[i, j]
            end
        end
    end
    return P_dict
end

function lengthRouteVec(linkCostDic)
    routeCostVec = Float64[]

    the_file = open("../temp_files/path_link_incidence_MA_journal.txt", "r")

    while (line=readline(the_file)) != ""
        if contains(line, "->")
            node_list = Int64[]
            link_list = String[]
            for i in split(line[1:end-1], "->")
                push!(node_list, int64(i))
            end
            for i in 1:length(node_list)-1
                push!(link_list, "$(node_list[i]),$(node_list[i+1])")
            end

    #         link_label_dict_[link]
            routeCost = sum([linkCostDic[string(link_label_dict_[link])] for link in link_list])
            push!(routeCostVec, routeCost)
    #         println(line)
    #         println(split(line[1:end-1], "->"))

    #         println(node_list)
    #         println(link_list)
    #         break
        end
    end  
    return routeCostVec
end

# obtain further info of the network

using JSON

function furInfo()

    link_label_dict = readstring(out_dir * files_ID * "/link_label_dict.json")
    link_label_dict = JSON.parse(link_label_dict)

    link_label_dict_ = readstring(out_dir * files_ID * "/link_label_dict_.json")
    link_label_dict_ = JSON.parse(link_label_dict_)

    link_length_dict = readstring(out_dir * files_ID * "/link_length_dict.json")
    link_length_dict = JSON.parse(link_length_dict)
    
    return link_label_dict, link_label_dict_, link_length_dict
end