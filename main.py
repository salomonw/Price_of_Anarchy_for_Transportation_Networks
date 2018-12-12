# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:50:19 2018

@author: Salomon Wollenstein
"""


# ---------------------------- Load Function and parameters ------------------
import os
os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')

execfile('load_fcn_par.py')

# ---------------------------- Preprocessing ----------------------------------

G = import_jing_net(dir_shpfile, files_ID, out_dir)

#G = zload(out_dir + 'G' + files_ID + '.pkz')

extract_2012_INRIX_data(dir_data , out_dir, filtered_data_dir, files_ID , confidence_score_min, c_value_min, year)

#filter_TMC_mult_files(dir_data, files_ID, confidence_score_min, c_value_min, out_dir, filtered_data_dir)

filter_dates_and_free_flow_calc(filtered_data_dir, files_ID, out_dir, percentile_free_flow, dates_input)

capacity_data(dir_capacity_data, files_ID, out_dir)

calculate_ref_speed_tables(out_dir, files_ID)

filter_time_instances(out_dir, files_ID, time_instances, data_granularity)

G_ = calculate_data_flows(out_dir, files_ID, time_instances, days_of_week)

# ------------------------------- OD Demand  ----------------------------------
# these algorithms runs for whole months, for example apr, we encourage to treat monthly timeframes beyond this point
# a future implementation can be easly done to overcome this problem 

od_pair_definition(out_dir, files_ID )

path_incidence_matrix(out_dir, files_ID, time_instances, number_of_routes_per_od, theta, lower_bound_route )


"RUN JULIA: GLS_j.jl "
#runGLS_f(out_dir, files_ID, time_instances, month_w, week_day_list, average_over_time)

# ------------------------------ Inverse Optimization -------------------------

parse_data_for_Julia(out_dir, files_ID, time_instances)

'''
RUN JULIA: 01. InvOpt_salo_f.jl  JULIA 0.6.4
'''

create_testing_set(week_day_list_1, week_day_list_2, week_day_list_3, year, month, time_instances, month_w, out_dir, files_ID)

week_day_list.append("all")

create_East_Massachusetts_trips(out_dir, files_ID, month_w, time_instances, n_zones, week_day_list)

create_East_Massachusetts_net(out_dir, files_ID, month_w, month, year, time_instances, n_zones, week_day_list )


month_id = month_w


execfile('../demand_adjustment_MA/01_find_paths.py')
#execfile('../demand_adjustment_MA/02_create_node_link_incidence.py')

execfile('../demand_adjustment_MA_density/03_create_node_link_incidence.py')


'''

RUN JULIA: POA_iteration

'''

















'''

RUN JULIA: 04. prepare_demand_data
include()

include("06.\ MSAcross.jl")

RUN JULIA: 05. demand_adjustment_full


'''









    
'''
RUN JULIA: 02. uni-class_traffic_assignment_MSA_function.jl         
'''

calc_testing_errors(out_dir, files_ID, time_instances, month_w, week_day_list, deg_grid, c_grid, lamb_grid, train_idx)

'''
RUN JULIA: 03. Plot_comparison_results_Apr.ipynb
'''
'''
RUN JULIA: 04. sensitivity_analysis_Finite_Difference_Approximation
'''


path_incidence_matrix_jing(out_dir, files_ID, time_instances, month_w, number_of_routes_per_od, theta, lower_bound_route )
create_node_link_incidence(out_dir, files_ID, time_instances)

'''RUN JULIA prepare_demand_data '''

parse_data_for_TAP(out_dir, files_ID, time_instances, month_w, week_day_list)

'''
RUN JULIA: 04.1 
'''

'''
RUN JULIA:  05. TAP POA.jl
'''
