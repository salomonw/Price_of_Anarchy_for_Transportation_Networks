# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:50:19 2018

@author: Salomon Wollenstein
"""

# ------------ Load libraries, functions and parameters -----------------------

import os

os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')

get_ipython().magic(u'run utils.py')

get_ipython().magic(u'run parameters.py')

get_ipython().magic(u'run import_jing_net.py')

get_ipython().magic(u'run functions.py')

get_ipython().magic(u'run OD_functions.py')

get_ipython().magic(u'run python_to_julia_fctn.py')

get_ipython().magic(u'run POA.py')

get_ipython().magic(u'run unzip_files.py')

# ---------------------------- Preprocessing ----------------------------------

G = import_jing_net(dir_shpfile, files_ID, out_dir)

#G = zload(out_dir + 'G' + files_ID + '.pkz')

extract_2012_INRIX_data(dir_data , out_dir, filtered_data_dir, files_ID , confidence_score_min, c_value_min)

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

runGLS(out_dir, files_ID, time_instances, 'Apr') 

# ------------------------------ Inverse Optimization -------------------------

parse_data_for_Julia(out_dir, files_ID, time_instances)

'''
RUN JULIA: 01. InvOpt_salo_f.jl  JULIA 0.6.4
'''
# testing sets
month_w = 'Apr'
month = 4
year = 2012
n_zones = 8
week_day_list_1 = [20, 23, 24, 25, 26, 27, 30]
week_day_list_2 = [11, 12, 13, 16, 17, 18, 19] 
week_day_list_3 = [2, 3, 4, 5, 6, 9, 10]

create_testing_set(week_day_list_1, week_day_list_2, week_day_list_3, year, month, time_instances, month_w, out_dir, files_ID)

create_East_Massachusetts_trips(out_dir, files_ID, month_w, time_instances, n_zones)

create_East_Massachusetts_net(out_dir, files_ID, month_w, month, year, time_instances, n_zones )

    
'''
RUN JULIA: 02. uni-class_traffic_assignment_MSA_function.jl         
'''
'''
RUN JULIA: 03. Plot_comparison_results_Apr.ipynb
'''
'''
RUN JULIA: 04. sensitivity_analysis_Finite_Difference_Approximation
'''

parse_data_for_TAP(out_dir, files_ID, time_instances, month_w)


'''
RUN JULIA:  05. TAP POA.jl
'''

InverseVI_uni_MA_with_base_trans_python(out_dir, files_ID, time_instances, month_w)



















calculate_testing_errors(out_dir, files_ID, month_w, instance, deg_grid, c_grid, lamb_grid):



month_w = 'Apr'
instance = 'AM'

import json
with open(out_dir + "/data_traffic_assignment_uni-class/" + files_ID + '_net_' + month_w + '_' + instance + '.txt') as MA_flow:
    MA_flow_lines = MA_flow.readlines()
MA_links = []
i = -9
for line in MA_flow_lines:
    
    i += 1
    if i > 0:
        MA_links.append(line.split()[1:3])
numLinks = i

link_list_js = [str(int(MA_links[i][0])) + ',' + str(int(float(MA_links[i][1]))) for \
             i in range(len(MA_links))]

link_list_pk = [str(int(MA_links[i][0])) + '->' + str(int(float(MA_links[i][1]))) for \
             i in range(len(MA_links))]

zdump(link_list_js, out_dir +  files_ID '_link_list_js.pkz')
zdump(link_list_pk, out_dir +  files_ID '_link_list_pk.pkz')
numNodes = max([int(float(MA_links[i][1])) for i in range(numLinks)])



#Compute Jacobian
import json
with open(out_dir + "/data_traffic_assignment_uni-class/" + files_ID + '_net_' + month_w + '_' + instance + '.txt') as MA_flow:
    MA_flow_lines = MA_flow.readlines()
MA_links = []
i = -9
for line in MA_flow_lines:
    i += 1
    if i > 0:
        MA_links.append(line.split('  ')[1:3])
numLinks = i

link_list_js = [str(int(MA_links[i][0])) + ',' + str(int(MA_links[i][1])) for \
                i in range(len(MA_links))]

link_list_pk = [str(int(MA_links[i][0])) + '->' + str(int(MA_links[i][1])) for \
                i in range(len(MA_links))]

numNodes = max([int(MA_links[i][1]) for i in range(numLinks)])

from collections import defaultdict

node_neighbors_dict = defaultdict(list)

for node in range(numNodes):
    for link in MA_links:
        if node == int(link[0]):
            node_neighbors_dict[str(node)].append(int(link[1]))

with open(out_dir + "/data_traffic_assignment_uni-class/" + files_ID + '_trips_' + month_w + '_' + instance + '.txt') as MA_trips:
    MA_trips_lines = MA_trips.readlines()

numZones = int(MA_trips_lines[0].split(' ')[3])

od_pairs = zload(out_dir + 'od_pairs' + files_ID + '.pkz')













import json

os.chdir(jing_folders[0])

link_label_dict_ = zload('link_label_dict_.pkz')

with open(out_dir + 'coeffs_dict_Apr_AM.json', 'r') as json_file:
    coeffs_dict_Apr_AM= json.load(json_file)


with open(out_dir + 'coeffs_dict_Apr_MD.json', 'r') as json_file:
    coeffs_dict_Apr_MD= json.load(json_file)


with open(out_dir + 'coeffs_dict_Apr_PM.json', 'r') as json_file:
    coeffs_dict_Apr_PM= json.load(json_file)


with open(out_dir + 'coeffs_dict_Apr_NT.json', 'r') as json_file:
    coeffs_dict_Apr_NT= json.load(json_file)


uni-class_traffic_assignment_MSA_flows_Apr_PM



with open(out_dir + 'uni-class_traffic_assignment_MSA_flows_Apr_PM.json', 'r') as json_file:
    uni_class_traffic_assignment_MSA_flows_Apr_PM = json.load(json_file)

















