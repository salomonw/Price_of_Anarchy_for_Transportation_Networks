# based on https://github.com/chkwon/TrafficAssignment.jl

include("Julia_files/load_network_uni_class.jl")

using Graphs

function create_graph(start_node, end_node)
    @assert Base.length(start_node)==Base.length(end_node)

    no_node = max(maximum(start_node), maximum(end_node))
    no_arc = Base.length(start_node)

    graph = simple_inclist(no_node)
    for i=1:no_arc
        add_edge!(graph, start_node[i], end_node[i])
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


function salo(out_dir, files_ID, month_w, week_day_list, instance, deg_grid, c_grid, lamb_grid)
    if isdir(out_dir * "uni-class_traffic_assignment") == false
        mkdir(out_dir * "uni-class_traffic_assignment")
    end

    instance1 = instance

    for day in week_day_list
        ta_data = load_ta_network_(out_dir, files_ID, month_w, day, instance1)

        # unpacking data from ta_data
        network_name = ta_data.network_name

        number_of_zones = ta_data.number_of_zones
        number_of_nodes = ta_data.number_of_nodes
        first_thru_node = ta_data.first_thru_node
        number_of_links = ta_data.number_of_links

        start_node = ta_data.start_node
        end_node = ta_data.end_node
        capacity = ta_data.capacity
        link_length = ta_data.link_length

        free_flow_time = ta_data.free_flow_time
        B = ta_data.B
        power = ta_data.power
        speed_limit = ta_data.speed_limit
        toll = ta_data.toll
        link_type = ta_data.link_type
        number_of_zones = ta_data.number_of_zones
        total_od_flow = ta_data.total_od_flow
        travel_demand = ta_data.travel_demand
        od_pairs = ta_data.od_pairs

        toll_factor = ta_data.toll_factor
        distance_factor = ta_data.distance_factor

        best_objective = ta_data.best_objective

        # preparing a graph
        graph = create_graph(start_node, end_node)


        link_dic = sparse(start_node, end_node, 1:number_of_links);

        function MSA(coeffs) 
            polyEval(coeffs, pt) = sum([coeffs[i] * pt^(i-1) for i = 1:length(coeffs)]) 

            function BPR(x)
                bpr = similar(x)
                for i=1:length(bpr)
                    bpr[i] = free_flow_time[i] * polyEval( coeffs, (x[i]/capacity[i]) ) 
                end
                return bpr
            end

            function all_or_nothing(travel_time)
                state = []
                path = []
                x = zeros(size(start_node))

                for r=1:size(travel_demand)[1]
                    # for each origin node r, find shortest paths to all destination nodes
                    state = dijkstra_shortest_paths(graph, travel_time, r)

                    for s=1:size(travel_demand)[2]
                        # for each destination node s, find the shortest-path vector
                        # load travel demand
                        x = x + travel_demand[r,s] * get_vector(state, r, s, link_dic)
                    end
                end

                return x
            end

            # Finding a starting feasible solution
            travel_time = BPR(zeros(number_of_links))
            xl = all_or_nothing(travel_time)

            max_iter_no = 1e3
            l = 1
            #average_excess_cost = 1
            tol = 1e-5

            while l < max_iter_no
                l += 1

                xl_old = xl

                # Finding yl
                travel_time = BPR(xl)

                yl = all_or_nothing(travel_time)

                xl = xl + (yl - xl)/l

                xl_new = xl

                relative_gap = norm(xl_new - xl_old, 1) / norm(xl_new, 1)

                if relative_gap < tol
                    break
                end
            end
            
            return xl
        end


        coeffs_dict_Apr_AM = 0
        # getting the coefficients of the costs
        coeffs_dict_Apr_AM = readstring(out_dir * "coeffs_dict_" * month_w * "_" * instance1 * ".json")
        coeffs_dict_Apr_AM = JSON.parse(coeffs_dict_Apr_AM)

        di = Dict()

        lenDeg = length(deg_grid)
        cnt = 0
        for deg in deg_grid
            for c in c_grid
                for lam in lamb_grid
                   # print("($(deg),$(c),$(lam),1)")
                    coeffs_1 = coeffs_dict_Apr_AM["($(deg), $(c), $(lam), 1)"]
                    coeffs_2 = coeffs_dict_Apr_AM["($(deg), $(c), $(lam), 2)"]
                    coeffs_3 = coeffs_dict_Apr_AM["($(deg), $(c), $(lam), 3)"]
                    ala = "($(deg), $(c), $(lam), $(1))"
                    apa = coeffs_3
                    #println(string(ala, apa))
                    di[(deg, c, lam, 1)]  = MSA(coeffs_1)
                    di[(deg, c, lam, 2)]  = MSA(coeffs_2)
                    di[(deg, c, lam, 3)]  = MSA(coeffs_3)
                end
            end
            cnt = cnt + 1 
            println("processed $(cnt) out of $(lenDeg)")
        end
        
        outfile = 0
        outfile = open(out_dir * "uni-class_traffic_assignment/MSA_flows_" * month_w * "_" * string(day) * '_' * instance1 * ".json", "w")
        JSON.print(outfile, di)
        close(outfile)
    end
end

#Importing parameters
using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "");
@pyimport parameters_julia

out_dir = parameters_julia.out_dir
files_ID = parameters_julia.files_ID
month_w = parameters_julia.month_w
year = parameters_julia.year
instances_1 = parameters_julia.instances_ID
deg_grid = parameters_julia.deg_grid
c_grid = parameters_julia.c_grid
lamb_grid = parameters_julia.lamb_grid
week_day_list = parameters_julia.week_day_list

for ins in instances_1
    salo(out_dir, files_ID, month_w, week_day_list, ins, deg_grid, c_grid, lamb_grid) #idx in length(instances_1)
end