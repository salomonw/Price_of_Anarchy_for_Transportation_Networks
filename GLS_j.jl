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

@pyimport OD_functions
@pyimport runGLS_f

out_dir = parameters_julia.out_dir;
files_ID = parameters_julia.files_ID;
month_w = parameters_julia.month_w;
year = parameters_julia.year;
time_instances = parameters_julia.instances_ID;
deg_grid = parameters_julia.deg_grid;
c_grid = parameters_julia.c_grid;
lamb_grid = parameters_julia.lamb_grid;
week_day_list = parameters_julia.week_day_list;
average_over_time_p = parameters_julia.average_over_time_p;
od_pairs = parameters_julia.od_pairs;


using JuMP, Ipopt
function GLS_p2(xi_list, P)
	mGLSJulia = Model(solver=IpoptSolver(max_iter=5000))

	@variable(mGLSJulia, lam[1:size(P,1)] >= 0)
	@variable(mGLSJulia, p[1:size(P,1), 1:size(P,2)] >= 0)

	for i = 1:size(P,1)
	    for j = 1:size(P,2)
	        if P[i,j] == 0
	            @constraint(mGLSJulia, p[i,j] == 0)
	        end
	    end
	end
	            
	for i = 1:size(P,1)
	    @constraint(mGLSJulia, sum{p[i,j], j = 1:size(P,2)} == 1)
	end

	for l = 1:size(P,2)
	    @NLconstraint(mGLSJulia, sum{p[i,l] * lam[i], i = 1:size(P,1)} == xi_list[l])
	end
	
	@objective(mGLSJulia, Min, sum{lam[j], j = 1:size(P,1)})  # play no actual role, but could not use zero objective
	#@NLobjective(mGLSJulia, Min, obj)
	#@NLobjective(mGLSJulia, Min, sum{ (sum{p[i,l] * lam[i], i = 1:size(P,1)} - xi_list[l])^2, l=1:size(P,1)})
	solve(mGLSJulia)

	return getvalue(lam), getvalue(p), getobjectivevalue(mGLSJulia)

end

function saveDemandVec(od_pairs, file_name, lam_list)
	lam = Dict()
	idx = 0
	open(file_name, "w") do file
		for od  = 1: size(od_pairs,1)
			i = od_pairs[od,1]
			j = od_pairs[od,2]
			idx += 1
			lam[idx] = lam_list[idx]
			dem = lam_list[idx]
			write(file, "$i, $j, $dem")#,"\n")
			write(file, "\n")
			#for ln =1:size(lam,1)
			#	println("$(i),$(j),$(lam_list[idx])")
			#end
		end
	close(file)
	end
end

#=
# Solve P1 and P2 of GLS for every day and every instance
for instance in time_instances
	for day in week_day_list
		xi_list, x, A, P, gls_cost = runGLS_f.GLS_p1(instance, day, average_over_time_p);
		P[P.>0] = 1;
		lam_, Pr, obj  = GLS_p2(xi_list, P);
		file_name = out_dir * "OD_demands/OD_demand_matrix_" * month_w * "_" * string(day) *  "_weekday_" * instance * files_ID * ".txt"
		saveDemandVec(od_pairs, file_name, lam_)
		P_file_name = out_dir * "OD_demands/OD_route_prob_matrix_" * month_w * "_" * string(day) *  "_weekday_" * instance * files_ID * ".txt"
		writedlm(P_file_name, Pr) 
		# Calculate flows APg
		flow_file = out_dir * "OD_demands/flow_APg_" * month_w * "_" * string(day) *  "_weekday_" * instance * files_ID * ".txt"
		writedlm(flow_file, A*Pr'*lam_ )
		# Store data flows flows 
		flow_file = out_dir * "OD_demands/flow_data_" * month_w * "_" * string(day) *  "_weekday_" * instance * files_ID * ".txt"
		writedlm(flow_file, mean(x,2) )
		#gls_name = out_dir * "OD_demands/gls_cost_vec_" * month_w * "_weekday_" * instance * files_ID * ".json"
		#outfile = open(gls_name, "w")
	    #JSON.print(outfile, gls_cost)
	    #close(outfile)
	end
end
=#

# Solve P1 and P2 of GLS for every instance using ALL observations
for instance in time_instances
	xi_list, x, A, P, gls_cost = runGLS_f.GLS_p1_all(instance, week_day_list, average_over_time_p, "all");
	P[P.>0] = 1;
	lam_, Pr, obj  = GLS_p2(xi_list, P);

	for i in ["all", "full"]
		file_name = out_dir * "OD_demands/OD_demand_matrix_" * month_w * "_" * i *  "_weekday_" * instance * files_ID * ".txt"
		saveDemandVec(od_pairs, file_name, lam_)

		P_file_name = out_dir * "OD_demands/OD_route_prob_matrix_" * month_w * "_" * i *  "_weekday_" * instance * files_ID * ".txt"
		writedlm(P_file_name, Pr) 

		# Calculate flows APg
		flow_file = out_dir * "OD_demands/flow_APg_" * month_w * "_" * i *  "_weekday_" * instance * files_ID * ".txt"
		writedlm(flow_file, A*Pr'*lam_ )

		# Store data flows flows 
		flow_file = out_dir * "OD_demands/flow_data_" * month_w * "_" * i *  "_weekday_" * instance * files_ID * ".txt"
		writedlm(flow_file, mean(x,2) )

		gls_name = out_dir * "OD_demands/gls_cost_vec_" * month_w * "_weekday_" * instance * files_ID * ".json"
		outfile = open(gls_name, "w")
	    JSON.print(outfile, gls_cost)
	    close(outfile)
	end
end
#=
# Solve P1 and P2 of GLS for every instance using daily averages
for instance in time_instances
	xi_list, x, A, P = runGLS_f.GLS_p1_all(instance, week_day_list, average_over_time_p, "all")
	P[P.>0] = 1;
	lam_, Pr, obj  = GLS_p2(xi_list, P);

	file_name = out_dir * "OD_demands/OD_demand_matrix_" * month_w * "_" * "full" *  "_weekday_" * instance * files_ID * ".txt"
	saveDemandVec(od_pairs, file_name, lam_)

	file_name = out_dir * "OD_demands/OD_demand_matrix_" * month_w * "_" * "all" *  "_weekday_" * instance * files_ID * ".txt"
	saveDemandVec(od_pairs, file_name, lam_)

end
=#