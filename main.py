# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:50:19 2018

@author: Salomon Wollenstein
"""



# ---------------------------- Preprocessing ----------------------------------

G = import_jing_net(dir_shpfile, files_ID, out_dir)

#G = zload(out_dir + 'G' + files_ID + '.pkz')

extract_2012_INRIX_data(dir_data , out_dir, filtered_data_dir, files_ID , confidence_score_min, c_value_min, year)

#filter_TMC_mult_files(dir_data, files_ID, confidence_score_min, c_value_min, out_dir)

filter_dates_and_free_flow_calc(filtered_data_dir, files_ID, out_dir, percentile_free_flow, dates_input)

capacity_data(dir_capacity_data, files_ID, out_dir)

calculate_ref_speed_tables(out_dir, files_ID)

filter_time_instances(out_dir, files_ID, time_instances, data_granularity)

G_ = calculate_data_flows(out_dir, files_ID, time_instances, days_of_week)

# ------------------------------- OD Demand  ----------------------------------
# this algorithms runs for whole months, for example apr, we encourage to treat monthly timeframes beyond this point
# a future implementation can be easly done to overcome this problem 

od_pair_definition(out_dir, files_ID )

path_incidence_matrix(out_dir, files_ID, time_instances, number_of_routes_per_od, theta, lower_bound_route )

runGLS_f(out_dir, files_ID, time_instances, month_w, week_day_list, average_over_time)

# ------------------------------ Inverse Optimization -------------------------

parse_data_for_Julia(out_dir, files_ID, time_instances)

'''
RUN JULIA: 01. InvOpt_salo_f.jl  JULIA 0.6.4
'''

create_testing_set(week_day_list_1, week_day_list_2, week_day_list_3, year, month, time_instances, month_w, out_dir, files_ID)

create_East_Massachusetts_trips(out_dir, files_ID, month_w, time_instances, n_zones, week_day_list)

create_East_Massachusetts_net(out_dir, files_ID, month_w, month, year, time_instances, n_zones, week_day_list )



    
'''
RUN JULIA: 02. uni-class_traffic_assignment_MSA_function.jl         
'''
'''
RUN JULIA: 03. Plot_comparison_results_Apr.ipynb
'''
'''
RUN JULIA: 04. sensitivity_analysis_Finite_Difference_Approximation
'''

parse_data_for_TAP(out_dir, files_ID, time_instances, month_w, week_day_list)

'''
RUN JULIA: 04.1 
'''

'''
RUN JULIA:  05. TAP POA.jl
'''

InverseVI_uni_MA_with_base_trans_python(out_dir, files_ID, time_instances, month_w)

def plot_POA(time_instances, out_dir, month_w):
	from utils import *
	import json
	import numpy as np
	iport os 
	import matplotlib.pyplot as plt
	for instance in time_instances['id']:
		with open(out_dir + "PoA_dict_noAdj_" + month_w + '_' + instance + '.json', 'r') as json_file:
			PoA_dict_noAdj = json.load(json_file)

		with open(out_dir + "PoA_dict_" + month_w + '_' + instance + '.json', 'r') as json_file:
			PoA_dict_ = json.load(json_file)

		  
		plt.legend([PoA_dict_, PoA_dict_noAdj], ["PoA", "PoA demand adj"], loc=0)
		plt.xlabel('Days of ' + month_w)
		plt.ylabel('PoA')
		#pylab.xlim(-0.1, 1.6)
		#pylab.ylim(0.9, 2.0)
		grid("on")
		savefig(out_dir + 'PoA'+'_' + instance + '_' + month_w +'.eps')


