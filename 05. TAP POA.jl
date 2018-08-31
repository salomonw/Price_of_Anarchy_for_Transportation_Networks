#Importing parameters
using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "");
@pyimport matplotlib.pyplot as plt
@pyimport numpy as np
@pyimport os
@pyimport pickle
@pyimport pandas as pd
@pyimport collections
@pyimport parameters_julia

out_dir = parameters_julia.out_dir
files_ID = parameters_julia.files_ID
month_w = parameters_julia.month_w
year = parameters_julia.year
instances_ = parameters_julia.instances_ID


instance_ = instances_[4];

using JSON

instance1 = instance_

open(out_dir * "instance_comm.txt", "w") do f
    write(f, instance1)
end

week_day_Apr_list = parameters_julia.week_day_list
#day = week_day_Apr_list[1]

using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "");
@pyimport utils_julia
@pyimport GLS_julia


using JuMP, Ipopt
cnt = 0
poaDict = Dict{}()
social = Dict{}()
user = Dict{}()
flow_user  = GLS_julia.GLS_juliaf();

#load OD pair-route incidence
odPairRoute = readstring(out_dir * "od_pair_route_incidence_" * instance1 *  files_ID * ".json");
odPairRoute = JSON.parse(odPairRoute);

#load link-route incidence
linkRoute = readstring(out_dir * "link_route_incidence_" * instance1 *  files_ID * ".json");
linkRoute = JSON.parse(linkRoute);

#load OD pair labels
odPairLabel = readstring(out_dir *  "od_pair_label_dict_refined.json");
odPairLabel = JSON.parse(odPairLabel);

odPairLabel_ = readstring(out_dir  *"od_pair_label_dict__refined.json");
odPairLabel_ = JSON.parse(odPairLabel_);

#load link labels
linkLabel = readstring(out_dir * "link_label_dict.json");
linkLabel = JSON.parse(linkLabel);

linkLabel_ = readstring(out_dir * "link_label_dict_.json");
linkLabel_ = JSON.parse(linkLabel_);

#load node-link incidence
nodeLink = readstring(out_dir * "node_link_incidence.json");
nodeLink = JSON.parse(nodeLink);

include("Julia_files/load_network_uni_class.jl");

for day in week_day_Apr_list


	ta_data = load_ta_network_(out_dir, files_ID, month_w, string(day) , instance1);

	capacity = ta_data.capacity;
	free_flow_time = ta_data.free_flow_time;

	############
	#Read in the demand file
	file = open(out_dir * "data_traffic_assignment_uni-class/" * files_ID * "_trips_" * month_w * "_" * string(day) * "_" * instance1 * ".txt")
	demands = Dict()
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
	close(file);

	demandsVec = zeros(length(odPairLabel_))

	for i = 1:length(demandsVec)
	    demandsVec[i] = demands[odPairLabel_["$i"][1], odPairLabel_["$i"][2]]
	end

	for key=keys(odPairRoute)
	    if contains(key, "56-")
	        println(key)
	    end
	end


	coeffs_dict_Apr_PM_ = readstring(out_dir * "coeffs_dict_" * month_w *  "_" * instance1 * ".json")
	coeffs_dict_Apr_PM_ = JSON.parse(coeffs_dict_Apr_PM_)
	fcoeffs = coeffs_dict_Apr_PM_["(7, 0.5, 1000.0, 1)"]
	polyDeg = length(fcoeffs)

	

	# m = Model(solver=GurobiSolver(OutputFlag=false))
	m = Model(solver=IpoptSolver());

	numLinks = length(linkLabel_)
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


	#@expression(m, f, sum{free_flow_time[a]*linkFlow[a] + .03*free_flow_time[a]*((linkFlow[a])^5)/((capacity[a])^4), a = 1:numLinks} )


	#@NLexpression(m, f, sum{ free_flow_time[a] * fcoeffs[i]  *linkFlow[a]^i / capacity[a]^(i-1) , i = 1:polyDeg , a = 1:numLinks }) ;

	@NLexpression(m, f, sum{free_flow_time[a] * fcoeffs[1] * linkFlow[a] +
	        free_flow_time[a] * fcoeffs[2] * linkFlow[a]^2 / capacity[a] +
	        free_flow_time[a] * fcoeffs[3] * linkFlow[a]^3 / capacity[a]^2 +
	        free_flow_time[a] * fcoeffs[4] * linkFlow[a]^4 / capacity[a]^3 +
	        free_flow_time[a] * fcoeffs[5] * linkFlow[a]^5 / capacity[a]^4 +
	        free_flow_time[a] * fcoeffs[6] * linkFlow[a]^6 / capacity[a]^5 +
			free_flow_time[a] * fcoeffs[7] * linkFlow[a]^7 / capacity[a]^6 , a = 1:numLinks})

	@NLobjective(m, Min, f);
	#print(m) 

	solve(m);

	flows = Dict();

	for i = 1:length(ta_data.start_node)
	    key = (ta_data.start_node[i], ta_data.end_node[i]);
	    flows[key] = getvalue(linkFlow)[i];
	end



	function socialObj(linkFlowVec)
	    objVal =  sum(sum(free_flow_time[a] * fcoeffs[i] * linkFlowVec[a]^i / capacity[a]^(i-1) for i=1:polyDeg) 
	        for a = 1:numLinks)
	    return objVal
	end
	



	PoA_dict = Dict();
	tapSocialFlowDicDict = Dict();
	tapSocialFlowVecDict = Dict();



	
	#cnt = 0
	#for i = 1:length(week_day_Apr_list)
	poaDictAprPM =  socialObj(flow_user[:,cnt+1])/getobjectivevalue(m)
	#println(socialObj(flow_user[:, cnt]))
	cnt +=1
	
	social[cnt] = getobjectivevalue(m)
	user[cnt] = socialObj(flow_user[:,cnt])
	poaDict[cnt] = user[cnt]/social[cnt]

end


user
social

poaDict
#=

=#