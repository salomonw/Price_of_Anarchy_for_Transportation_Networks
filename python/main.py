# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:50:19 2018

@author: Salomon Wollenstein
"""
import os 

os.chdir('C:/Users/Salomon Wollenstein/Documents/GitHub/PoA_Jing_net/python')


get_ipython().magic(u'run utils.py')

get_ipython().magic(u'run parameters.py')

get_ipython().magic(u'run import_jing_net.py')

get_ipython().magic(u'run functions.py')

get_ipython().magic(u'run unzip_files.py')

# ---------- Preprocessing -----------------------

G = import_jing_net(dir_shpfile, files_ID, out_dir) 
G = zload(out_dir + 'G' + files_ID + '.pkz')
    

extract_2012_INRIX_data(dir_data , out_dir, filtered_data_dir, files_ID , confidence_score_min,c_value_min)

#filter_TMC_mult_files(dir_data, files_ID, confidence_score_min, c_value_min, out_dir)

filter_dates_and_free_flow_calc(filtered_data_dir, files_ID, out_dir, percentile_free_flow, dates_input)

capacity_data(dir_capacity_data, files_ID, out_dir)

calculate_ref_speed_tables(out_dir, files_ID)

filter_time_instances(out_dir, files_ID, time_instances, data_granularity)

G_ = calculate_data_flows(out_dir, files_ID, time_instances, days_of_week)


# ---------- OD Demand  -----------------------
