# ------------------- CLASSES ------------------------
#include("defArc.jl")

type Arc
    initNode::Int
    termNode::Int
    capacity::Float64
    freeflowtime::Float64
    flow::Float64
end

Arc(initNode::Int, termNode::Int, capacity::Float64, freeflowtime::Float64) =
    Arc(initNode, termNode, capacity, freeflowtime, 0.)



#include("fitTraffic.jl")

## Solve an inverse tarffic problem over polynomials
## of degree at most d
## optionally use a regularizer from the poly kernel

using JuMP
using Gurobi
#using LinAlg

polyEval(coeffs, pt) = sum([coeffs[i] * pt^(i-1) for i = 1:length(coeffs)])

polyEval(coeffs::Array{Float64, 1}, pt) = sum([coeffs[i] * pt^(i-1) for i = 1:length(coeffs)])

bpacost(flow::Float64, capacity::Float64, freeflowtime::Float64) = freeflowtime*(1 + .15 * (flow/capacity)^4)
bpacost(flow::Float64, arc) = bpacost(flow, arc.capacity, arc.freeflowtime)
bpacost(arc::Arc) = bpacost(arc.flow, arc)

# --------------------  FUNCTIONS  -------------------

function setUpFitting(deg::Int, c::Float64)
    m = 0 
	m = Model(solver=GurobiSolver(OutputFlag=false))

	@variable(m, coeffs[1:deg+1])
	@variable(m, Calphas[1:deg+1])

	#build the graham matrix; cf. Ref. [21] (Regularization Networks and Support Vector Machines), page 47
	samples = linspace(0, 1, deg + 1)
	k(x,y) = (c + x*y)^deg
	K = [ k(x,y) for x = samples, y=samples]
	K = convert(Array{Float64, 2}, K)
	#assert(rank(K) == deg+1)

	C = chol(K + 1e-6* eye(deg+1))
	for i=1:deg + 1
		@constraint(m, polyEval(coeffs, samples[i]) == sum{C[j, i] * Calphas[j], j=1:deg+1})
	end

	@variable(m, reg_term >= 0)
	reg_term_ = QuadExpr(Calphas[:], Calphas[:], ones(deg+1), AffExpr())

	@constraint(m, reg_term >= reg_term_)

	return m, coeffs, reg_term
end



function fixCoeffs(m, fcoeffs, coeffs)
	for (fc, c) in zip(fcoeffs, coeffs[:])
		@constraint(m, fc == c)
	end
end



function addResid(m, coeffs, ys, demands, arcs, scaling)
	@variable(m, resid)
	@variable(m, dual_cost)
	@variable(m, primal_cost)

	@constraint(m, dual_cost == sum{demands[(s,t)] * (ys[(s,t), t] - ys[(s,t), s]), (s,t)=keys(demands)})
	@constraint(m, primal_cost == sum{a.flow * a.freeflowtime * polyEval(coeffs, a.flow/a.capacity), a=values(arcs)})

	@constraint(m, resid >= (dual_cost - primal_cost) / scaling )
	@constraint(m, resid >= (primal_cost - dual_cost) / scaling )
	return resid
end



function addIncreasingCnsts(m, coeffs, arcs; TOL=0.)
	sorted_flows = sort([a.flow / a.capacity for a in values(arcs)])
	@constraint(m, polyEval(coeffs, 0) <= polyEval(coeffs, sorted_flows[1]))
	for i = 2:length(sorted_flows)
		@constraint(m, polyEval(coeffs, sorted_flows[i-1]) <= polyEval(coeffs, sorted_flows[i]) + TOL)
	end
    @constraint(m, coeffs[1] == 1)  # enforce g(0) = 1
end



#equates the total cost of the network to the true total cost
function normalize(m, coeffs, tot_true_cost::Float64, arcs)
	@constraint(m,
		sum{a.freeflowtime * a.flow * polyEval(coeffs, a.flow / a.capacity), a=values(arcs)} == tot_true_cost)
end


function normalize(m, coeffs, scaled_flow::Float64, cost::Float64)
	@constraint(m, polyEval(coeffs, scaled_flow) == cost)
end


function normalize(m, coeffs, scaled_flows::Array{Float64, 1}, avgCost::Float64)
    @constraint(m, sum{polyEval(coeffs, f), f=scaled_flows} == avgCost * length(scaled_flows))
end



function addNetworkCnsts(m, coeffs, demands, arcs, numNodes)
	@variable(m, ys[keys(demands), 1:numNodes])
	for k = keys(arcs)
		a = arcs[k]
		rhs = a.freeflowtime * polyEval(coeffs, a.flow/a.capacity)
		for od in keys(demands)
			@constraint(m, ys[od, k[2]] - ys[od, k[1]] <= rhs)
		end
	end
	return ys
end



function read_demand_file(file_path)
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


using JSON
function read_flow_after_cons(file_path)

    flow_after_conservation = readstring(file_path);
    flow_after_conservation = replace(flow_after_conservation, "NaN", 0);
    flow_after_conservation = JSON.parse(flow_after_conservation);
    return flow_after_conservation
end



function read_link_day_min_dict(file_path)
    link_day_minute_Apr_dict = readstring(file_path);
    link_day_minute_Apr_dict = replace(link_day_minute_Apr_dict, "NaN", 0);
    link_day_minute_Apr_dict = JSON.parse(link_day_minute_Apr_dict);
    return link_day_minute_Apr_dict
end

##########
#Fitting Funcs
##########

function train(indices, lam::Float64, deg::Int, c::Float64, demand_data, flow_data, arcs, vArcs, demands; fcoeffs=nothing)
    numNodes = maximum(map(pair->pair[1], keys(arcs)))
    m, coeffs, reg_term = setUpFitting(deg, c)

    addIncreasingCnsts(m, coeffs, arcs, TOL=1e-8)  #uses the original obs flows

    avgCost = mean( [bpacost(a.flow, a.capacity, 1.0) for a in values(arcs)] )
    normalize(m, coeffs, [a.flow / a.capacity for a in values(arcs)], avgCost)


    resids = Variable[]

    for i = indices
        #copy the flow data over to the arcs, demand data to demands (slow)
        for (ix, a) in enumerate(vArcs)
            a.flow = flow_data[ix, i]
        end
        for odpair in keys(demands)
            demands[odpair] = demand_data[odpair][i]
        end

        #Dual Feasibility
        ys = addNetworkCnsts(m, coeffs, demands, arcs, numNodes)

        #add the residual for this data point
        push!(resids, addResid(m, coeffs, ys, demands, arcs, 1e6))
    end

    if fcoeffs != nothing
        fixCoeffs(m, fcoeffs, coeffs)
    end
    @objective(m, Min, sum{resids[i], i = 1:length(resids)} + lam*reg_term)
    solve(m)
    #println(getObjectiveValue(m) - lam * getValue(reg_term) )
    return [getvalue(coeffs[i]) for i =1:length(coeffs)]
end


# --------------------  PARAMETERS -------------------

#Importing parameters
using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "");
@pyimport parameters_julia

out_dir = parameters_julia.out_dir
files_ID = parameters_julia.files_ID
month_w = parameters_julia.month_w
year = parameters_julia.year
insts = parameters_julia.instances_ID
deg_grid = parameters_julia.deg_grid
c_grid = parameters_julia.c_grid
lamb_grid = parameters_julia.lamb_grid
#println(deg_grid)
week_day_Apr_list_1 = parameters_julia.week_day_list_1
week_day_Apr_list_2 = parameters_julia.week_day_list_2
week_day_Apr_list_3 = parameters_julia.week_day_list_3


n_nodes = parameters_julia.n_zones
month = parameters_julia.month
month_w = parameters_julia.month_w
year = parameters_julia.year

using Graphs
#for inst in insts
    inst = "NT"
	# Assing file names 
	link_day_min_path = out_dir * "link_min_dict" * files_ID * ".json"
    #flow_path =  out_dir * "flows_after_QP" * files_ID * "_" * inst * ".json"
    flow_path =  out_dir * "density_links" * files_ID * "_" * inst * ".json"
    save_file_path = out_dir * "coeffs_dict_" * month_w * "_" * inst * ".json"
    OD_demand_path = out_dir * "OD_demands/OD_demand_matrix_" * month_w * "_" * "full" * "_weekday_" * inst * files_ID * ".txt"


    demands = read_demand_file(OD_demand_path)
    flow_after_conservation = read_flow_after_cons(flow_path)
    link_day_minute_Apr_dict = read_link_day_min_dict(link_day_min_path)
    key1 = [uppercase(key1) for key1 in keys(flow_after_conservation)]
    key2 = key1[1]
    arc_ids = [uppercase(ids) for ids in keys(flow_after_conservation[key2])]
    arcs_1 = Dict[]
    for j in week_day_Apr_list_1
        arcs_1_ = Dict()
        for i in arc_ids
            key = "link_$(i)_$(year)_$(month)_$(j)"
            initNode = link_day_minute_Apr_dict[key]["init_node"]
            termNode = link_day_minute_Apr_dict[key]["term_node"]
            capacity = link_day_minute_Apr_dict[key]["capac_$inst"]
            freeflowtime = link_day_minute_Apr_dict[key]["free_flow_time"]
            flow = link_day_minute_Apr_dict[key]["avg_flow_$inst"]
            arcs_1_[(initNode, termNode)] = Arc(initNode, termNode, capacity, freeflowtime, flow)
        end
        push!(arcs_1, arcs_1_)
    end

    arcs_2 = Dict[]
    for j in week_day_Apr_list_2
        arcs_2_ = Dict()
        for i in arc_ids
            key = "link_$(i)_$(year)_$(month)_$(j)"
            initNode = link_day_minute_Apr_dict[key]["init_node"]
            termNode = link_day_minute_Apr_dict[key]["term_node"]
            capacity = link_day_minute_Apr_dict[key]["capac_$inst"]
            freeflowtime = link_day_minute_Apr_dict[key]["free_flow_time"]
            flow = link_day_minute_Apr_dict[key]["avg_flow_$inst"]
            arcs_2_[(initNode, termNode)] = Arc(initNode, termNode, capacity, freeflowtime, flow)
        end
        push!(arcs_2, arcs_2_)
    end
    arcs_3 = Dict[]
    for j in week_day_Apr_list_3
        arcs_3_ = Dict()
        for i in arc_ids
            key = "link_$(i)_$(year)_$(month)_$(j)"
            initNode = link_day_minute_Apr_dict[key]["init_node"]
            termNode = link_day_minute_Apr_dict[key]["term_node"]
            capacity = link_day_minute_Apr_dict[key]["capac_$inst"]
            freeflowtime = link_day_minute_Apr_dict[key]["free_flow_time"]
            flow = link_day_minute_Apr_dict[key]["avg_flow_$inst"]
            arcs_3_[(initNode, termNode)] = Arc(initNode, termNode, capacity, freeflowtime, flow)
        end
        push!(arcs_3, arcs_3_)
    end
    ##########
    # Set up demand data and flow data
    ##########
    numData = length(arcs_1)
    sigma = .0
    flow_data_1 = Array(Float64, length(arcs_1[1]), numData)
    flow_data_2 = Array(Float64, length(arcs_2[1]), numData)
    flow_data_3 = Array(Float64, length(arcs_3[1]), numData)

    demand_data = Dict()

    numNodes = maximum(map(pair->pair[1], keys(demands)))
    g = simple_inclist(numNodes, is_directed=true)
    vArcs = Arc[]
    for arc in values(arcs_1[1])
        add_edge!(g, arc.initNode, arc.termNode)
        push!(vArcs, arc)
    end

    for iRun = 1:numData
        for odpair in keys(demands)
            if ! haskey(demand_data, odpair)
            
                demand_data[odpair] = [demands[odpair], ]
            else
                push!(demand_data[odpair], demands[odpair])
            end
        end

        #flow_data[:, iRun] = [a.flow::Float64 for a in vArcs]
        flow_data_1[:, iRun] = [a.flow::Float64 for a in values(arcs_1[iRun])]
        flow_data_2[:, iRun] = [a.flow::Float64 for a in values(arcs_2[iRun])]
        flow_data_3[:, iRun] = [a.flow::Float64 for a in values(arcs_3[iRun])]
    end

    # Train

    coeffs_dict_Apr_PM = Dict()
   # println(deg_grid)
    coeffs_keys = Vector()
   #coeffs_keys_2 = []
   #coeffs_keys_3 = []
    for deg in deg_grid
        for c in c_grid
            for lam in lamb_grid
                #println(numData)

                coeffs_dict_Apr_1 = train(1:numData,lam,deg,c,demand_data,flow_data_1,arcs_1[1], vArcs, demands)
                #println(coeffs_dict_Apr_1[1])
                #println(isa(coeffs_dict_Apr_1[1], Number))
                
                if isnan(coeffs_dict_Apr_1[1]) == false
                    coeffs_dict_Apr_PM[(deg,c,lam,1)] = coeffs_dict_Apr_1
                    append!(coeffs_keys, ["(" * string(deg) * ", " *  string(c) * ", " * string(lam) * ", 1)"])
                end
                
                coeffs_dict_Apr_2 = train(1:numData,lam,deg,c,demand_data,flow_data_1,arcs_2[1], vArcs, demands)
                if isnan(coeffs_dict_Apr_2[1]) == false
                    coeffs_dict_Apr_PM[(deg,c,lam,2)] = coeffs_dict_Apr_2
                    append!(coeffs_keys, ["(" * string(deg) * ", " *  string(c) * ", " * string(lam) * ", 2)"])
                end
                
                coeffs_dict_Apr_3 = train(1:numData,lam,deg,c,demand_data,flow_data_1,arcs_3[1], vArcs, demands)
                if isnan(coeffs_dict_Apr_3[1]) == false
                    coeffs_dict_Apr_PM[(deg,c,lam,3)] = coeffs_dict_Apr_3
                    append!(coeffs_keys, ["(" * string(deg) * ", " *  string(c) * ", " * string(lam) * ", 3)"])    
                end

                #coeffs_dict_Apr_2 = train(1:numData,lam,deg,c,demand_data,flow_data_2,arcs_1[1], vArcs, demands)
                #coeffs_dict_Apr_3 = train(1:numData,lam,deg,c,demand_data,flow_data_3,arcs_1[1], vArcs, demands)

                #coeffs_dict_Apr_PM[(deg,c,lam,2)] = train(1:numData,lam,deg,c,demand_data,flow_data_2,arcs_2[1], vArcs, demands)
                #coeffs_dict_Apr_PM[(deg,c,lam,3)] = train(1:numData,lam,deg,c,demand_data,flow_data_3,arcs_3[1], vArcs, demands)

                
            end
        end
    end

    outfile = open(save_file_path, "w")
    JSON.print(outfile, coeffs_dict_Apr_PM)
    close(outfile)

    outfile = open(out_dir * "coeffs_keys_" * month_w * "_" * inst * ".json", "w")
    JSON.print(outfile, coeffs_keys)
    close(outfile)
#end