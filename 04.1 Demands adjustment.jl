# https://github.com/jingzbu/InverseVIsTraffic/blob/master/08_develop_new_OD_demand_estimator_MA_uni_class_cdc16/06_demands_adjustment_MA.ipynb
# REMEMBER THAT THE FOLDER MUST BE EMPTY!!
#Importing parameters
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


out_dir = parameters_julia.out_dir
files_ID = parameters_julia.files_ID
month_w = parameters_julia.month_w
year = parameters_julia.year
instances_1 = parameters_julia.instances_ID
deg_grid = parameters_julia.deg_grid
c_grid = parameters_julia.c_grid
lamb_grid = parameters_julia.lamb_grid
week_day_Apr_list = parameters_julia.week_day_list


include("Julia_files/initia_data.jl");
include("prepare_data.jl");
#include("Julia_files/inverseVI.jl");
include("Julia_files/demands_adjustment_gradi.jl");

function demandsDictFixed(demandsDict, flow_observ, link_vector, graph, ta_data, link_dic, day, gamma1, 
    gamma2, out_dir, files_ID, month_w, instance, key_, free_flow_time, capacity, start_node, en_node, numZones, cnt)
    #day = 4  # day of April
    # observed flow vector
    xl = flow_observ[:, cnt]
    
    tapFlows = Dict()
    for i = 1:length(ta_data.start_node)
        key = link_vector[i]
        tapFlows[key] = xl[i]
    end
    #println(tapFlows)
   # tapFlowVect = xl;
    tapFlowVect = []
    for i = 1:length(start_node)
        key = link_vector[i]
        append!(tapFlowVect, tapFlows[key])
    end

    # get observed flow vector (corresponding to ground truth demands and ground truth costs)
    tapFlowDicDict[0], tapFlowVecDict[0] = tapFlows, tapFlowVect;

    # get arcs data corresponding to ground truth demands and flows !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    network_data_file = files_ID * "_net_" * month_w * "_full_" * instance * ".txt"
    arcsDict[0] = observFlow(out_dir  * "data_traffic_assignment_uni-class/" * network_data_file, tapFlowDicDict[0]);

    coeffs_dict_Apr_weekend_ = readstring(out_dir * "coeffs_dict_"* month_w * "_" * instance * ".json")
    coeffs_dict_Apr_weekend_ = JSON.parse(coeffs_dict_Apr_weekend_)
    fcoeffs = coeffs_dict_Apr_weekend_[key_]
    
    demandsVecDict[1] = demandsDicToVec(demandsDict[1]);
    #println(demandsVecDict[1])
    
    objFunDict[1] = objF(graph, ta_data, link_dic, gamma1, gamma2, demandsVecDict[1], demandsVecDict[1], tapFlowVecDict, fcoeffs, free_flow_time, capacity, start_node, en_node, numZones);

    # get initial flow vector (corresponding to initial demands)
    tapFlowDicDict[1], tapFlowVecDict[1] = tapMSA(graph, ta_data, link_dic, demandsDict[1], fcoeffs, free_flow_time, capacity, start_node, en_node, numZones)
    #tapMSA(graph, ta_data, link_dic, demandsDict[1], fcoeffs, free_flow_time, capacity, start_node, en_node,  numIter=500, tol=1e-6);       

    demandsVecDict[0] = demandsDicToVec(demandsDict[0]);

    # get arcs data corresponding to initial demands and flows !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    network_data_file = files_ID * "_net_" * month_w * "_full_" * instance * ".txt"
    arcsDict[1] = observFlow(out_dir  * "data_traffic_assignment_uni-class/" * network_data_file, tapFlowDicDict[1]);

    linkCostDicDict[1] = tapFlowVecToLinkCostDict(tapFlowVecDict[1], fcoeffs, free_flow_time, capacity);

    linkCostDicDict[1]["0"], link_length_list[1]

    jacobiSpiessDict[1] = Compute_Jacobian.jacobianSpiess(numNodes, numLinks, numODpairs, od_pairs,
                                                  link_list_js, [linkCostDicDict[1]["$(i)"] for i=0:numLinks-1]);

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

        descDirecDict[l] = descDirec(gamma1, gamma2, demandsVecDict[l], demandsVecDict[1],  tapFlowVecDict[l],
            tapFlowVecDict[0], jacobDict[l], numODpairs, numLinks);

        demandsVecDict[l] = demandsDicToVec(demandsDict[l]);

        searchDirecDict[l] = searchDirec(demandsVecDict[l], descDirecDict[l], epsilon_1);

        thetaMaxDict[l] = thetaMax(demandsVecDict[l], searchDirecDict[l]);

        demandsVecDict[l+1] = similar(demandsVecDict[0]);

        demandsVecDict[l+1], objFunDict[l+1] = armijo(gamma1, gamma2, objFunDict[l], demandsVecDict[l], demandsVecDict[0], tapFlowVecDict, fcoeffs, 
        	searchDirecDict[l], thetaMaxDict[l], rho, M, graph, ta_data, link_dic, free_flow_time, capacity, start_node, en_node, numZones);

        demandsDict[l+1] = demandsVecToDic(demandsVecDict[l+1]);

        tapFlowDicDict[l+1], tapFlowVecDict[l+1] = tapMSA(graph, ta_data, link_dic, demandsDict[l+1], fcoeffs, free_flow_time, capacity, start_node, en_node, numZones);

        arcsDict[l+1] = observFlow(out_dir  * "data_traffic_assignment_uni-class/" * network_data_file, tapFlowDicDict[l+1]);

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

    
    

    outfile = open(out_dir * "demandsDict/demandsVecDict$(day)_" * month_w * "_" * instance * ".json", "w")

    JSON.print(outfile, demandsVecDict)

    close(outfile)

    outfile = open(out_dir * "demandsDict/demandsDict$(day)_" * month_w * "_" * instance * ".json", "w")

    JSON.print(outfile, demandsDict)

    close(outfile)

    outfile = open(out_dir * "demandsDict/tapFlowDicDict$(day)_" * month_w * "_" * instance * ".json", "w")

    JSON.print(outfile, tapFlowDicDict)

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

    return obj

end

function socialObj(linkFlowVec, free_flow_time, polyDeg, fcoeffs, capacity, numLinks)
    objVal = sum([sum([free_flow_time[a] * fcoeffs[i] * linkFlowVec[a]^i / capacity[a]^(i-1) for i=1:polyDeg]) 
        for a = 1:numLinks])
    return objVal
end



#key_ = "(6, 1.5, 0.1, 1)"
instance = "AM"
instance1 = instance
open(out_dir * "instance_comm.txt", "w") do f
    write(f, instance1)
end

using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "");
@pyimport parameters_julia
@pyimport GLS_julia
@pyimport Compute_Jacobian

numNodes = Compute_Jacobian.numNodes;
numLinks = Compute_Jacobian.numLinks;
numODpairs = Compute_Jacobian.numODpairs;
numZones = Compute_Jacobian.numZones
od_pairs = Compute_Jacobian.od_pairs;
link_list_js = Compute_Jacobian.link_list_js;
link_length_list = Compute_Jacobian.link_length_list;


flow_observ, link_vector = GLS_julia.GLS_juliaf()

include("extract_data.jl");
include("Julia_files/tap_MSA.jl");
include("Julia_files/demands_adjustment_gradi.jl");

cnt = 0 
PoA_dict = Dict();
tapSocialFlowDicDict = Dict();
tapSocialFlowVecDict = Dict();
user_sol_dict = Dict();
social_sol_dict = Dict();
obj_dict = Dict();

for day in week_day_Apr_list
	cnt = cnt + 1
#day = 9
	demandsDict = extract_demandDict(out_dir, files_ID, month_w ,day, instance1, numZones)
	numNodes, numLinks, numODpairs, capacity, free_flow_time, ta_data_Apr_PM, start_node, en_node = extract_dataf(out_dir, files_ID, month_w ,day, instance1, demandsDict)

	# preparing a graph
	graph = create_graph(start_node, en_node);
	link_dic = sparse(start_node, en_node, 1:numLinks);
	
	demandsVecDict[0] = demandsDicToVec(demandsDict[0]);
	demandsDiffDict[1] = norm(demandsDicToVec(demandsDict[1]) - demandsDicToVec(demandsDict[0]))/
	                     norm(demandsDicToVec(demandsDict[0]));
	gamma1 = 0
	gamma2 = 100

	if isdir(out_dir * "demandsDict") == false
	    mkdir(out_dir * "demandsDict")
	end

    ta_data = load_ta_network_(out_dir, files_ID, month_w, day, instance1);
    
    key_ = readstring(out_dir * "cross_validation_best_key/cross_validation_best_key_" * month_w * "_" * string(day) * "_" * instance * ".json")
    key_ = JSON.parse(key_)
    #key_ = "(7, 0.5, 1000.0, 1)"
    obj_dict[day] = demandsDictFixed(demandsDict, flow_observ,link_vector, graph, ta_data, link_dic, day, gamma1, gamma2, 
                        out_dir, files_ID, month_w, instance, key_, free_flow_time, capacity, start_node, en_node, numZones, cnt)

    

	#end

	coeffs_dict_ = readstring(out_dir * "coeffs_dict_" * month_w * "_" * instance1 *".json")
	coeffs_dict_ = JSON.parse(coeffs_dict_)
	fcoeffs = coeffs_dict_[key_]
	polyDeg = length(fcoeffs)





	#for day in week_day_Apr_list

	demandsDict = readstring(out_dir * "demandsDict/demandsDictFixed$(day)_" * month_w * "_" * instance1 * ".json");
	demandsDict = JSON.parse(demandsDict)

	demandsDict_ = Dict()
	for key in keys(demandsDict)
	    key_2 = (parse(Int, split(split(key, ',')[1], '(')[2]), parse(Int, split(split(key, ',')[2], ')')[1]))
	    demandsDict_[key_2] = demandsDict[key]
	end

	#     demandsDict_

	     tapFlowDicDictP = Dict()
	     tapFlowVecDictP = Dict()
	     #tapFlowDicDict[day], tapFlowVecDict[day] = tapMSA(graph, ta_data, link_dic, demandsDict_, fcoeffs, free_flow_time, capacity, start_node, en_node, numZones)

	#     tapFlowVecDict[day]

	    tapSocialFlowDicDict[day], tapSocialFlowVecDict[day] = tapMSASocial(demandsDict_, fcoeffs, ta_data, graph, link_dic, start_node, en_node, free_flow_time, capacity, numLinks, numZones);
#demands, fcoeffs, graph, ta_data, link_dic, start_node, en_node, free_flow_time, capacity, numLinks, numIter=1000, tol=1e-6
	#     tapSocialFlowVecDict[day]

	#     flow_observ[:, day]

        tapFlowVecDictP = readstring(out_dir * "demandsDict/tapFlowVecDict$(day)_" * month_w * "_" * instance * ".json") 
        tapFlowVecDictP = JSON.parse(tapFlowVecDictP)
        k_ = keys(tapFlowVecDictP)
        max_k = maximum(map(x->parse(Int,x),k_))
        #println(max_k)
	    # PoA_dict[day] = socialObj(tapFlowVecDict[day]) / socialObj(tapSocialFlowVecDict[day])
        user_sol_dict[day] = socialObj(flow_observ[:, cnt], free_flow_time, polyDeg, fcoeffs, capacity, numLinks) ;
        #user_sol_dict[day] = socialObj(tapFlowVecDictP[string(max_k)], free_flow_time, polyDeg, fcoeffs, capacity, numLinks) ;
        social_sol_dict[day] = socialObj(tapSocialFlowVecDict[day], free_flow_time, polyDeg, fcoeffs, capacity, numLinks);
	    PoA_dict[day] = user_sol_dict[day] / social_sol_dict[day];
	#end
	println("day $(day) finished...")
    println("PoA for day $(day) and instance " * instance1 * " is " * string(PoA_dict[day]))
end


outfile =  open(out_dir * "PoA_dict_" * month_w * "_" * instance1 * ".json", "w")
JSON.print(outfile, PoA_dict)
close(outfile)


outfile = open(out_dir * "tapSocialFlowVecDict_" * month_w * "_" * instance1 * ".json", "w")
JSON.print(outfile, tapSocialFlowVecDict)
close(outfile)



outfile = open(out_dir * "cong_" * month_w * "_" * instance1 * ".json", "w")
JSON.print(outfile, user_sol_dict)
close(outfile)

outfile = open(out_dir * "obj_dict" * month_w * "_" * instance1 * ".json", "w")
JSON.print(outfile, obj_dict)
close(outfile)


#=

#for day = 1: length(week_day_Apr_list)

    demandsDict = readstring(out_dir * "demandsDict/demandsDictFixed$(day)_" * month_w * "_" * instance * ".json");
    demandsDict = JSON.parse(demandsDict)

    demandsDict_ = Dict()
    for key in keys(demandsDict)
        key_ = (parse(Int, split(split(key, ',')[1], '(')[2]), parse(Int, split(split(key, ',')[2], ')')[1]))
        demandsDict_[key_] = demandsDict[key]
    end

#     demandsDict_

#     tapFlowDicDict = Dict()
#     tapFlowVecDict = Dict()
#     tapFlowDicDict[day], tapFlowVecDict[day] = tapMSA(demandsDict_, fcoeffs);

#     tapFlowVecDict[day]

    tapSocialFlowDicDict[day], tapSocialFlowVecDict[day] = tapMSASocial(demandsDict_, fcoeffs);

#     tapSocialFlowVecDict[day]

#     flow_observ[:, day]

    # PoA_dict[day] = socialObj(tapFlowVecDict[day]) / socialObj(tapSocialFlowVecDict[day])

    PoA_dict[day] = socialObj(flow_observ[:, day]) / socialObj(tapSocialFlowVecDict[day]);
    println(socialObj(flow_observ[:, day]))
#end

#PoA_dict

=#

