using JSON
@pyimport parameters_julia
out_dir = parameters_julia.out_dir
files_ID = parameters_julia.files_ID
month_w = parameters_julia.month_w

instance1 = readstring(out_dir * "instance_comm.txt")

demandsDict = Dict()
# get ground trueth demands, indexed by 0
demandsDict[0] = iniDemand(out_dir * "data_traffic_assignment_uni-class/" * files_ID * "_trips_" * month_w * "_" * instance1 * ".txt")
# get initial demands, indexed by 1
demandsDict[1] = iniDemand(out_dir * "data_traffic_assignment_uni-class/" * files_ID * "_trips_" * month_w * "_" * instance1 * ".txt", 1)

numNodes, numLinks, numODpairs, capacity, free_flow_time, ta_data_Apr_PM = paraNetwork(out_dir, files_ID, month_w, instance1)
link_label_dict, link_label_dict_, link_length_dict = furInfo()

start_node = ta_data_Apr_PM.start_node
end_node = ta_data_Apr_PM.end_node

#load OD pair labels
odPairLabel = readstring(out_dir * "od_pair_label_dict_refined.json")
odPairLabel = JSON.parse(odPairLabel)

odPairLabel_ = readstring(out_dir * "od_pair_label_dict__refined.json")
odPairLabel_ = JSON.parse(odPairLabel_)

#load node-link incidence
nodeLink = readstring(out_dir * "node_link_incidence.json");
nodeLink = JSON.parse(nodeLink);