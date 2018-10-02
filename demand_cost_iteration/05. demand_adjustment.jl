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

include("load_Jfun.jl")


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
   # fcoeffs = [1,0,0,0,.5,0,0,0]
    #fcoeffs = [1.0,-0.00302509, 0.0577279, -0.195632, 0.620696, -0.905963, 0.936143, -0.469483, 0.108584]
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
