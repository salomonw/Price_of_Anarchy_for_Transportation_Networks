# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:50:08 2018

@author: Salomon Wollenstein
"""
import pandas as pd
import os 

'Parameters'
dir_shpfile = 'C:/Users/Salomon Wollenstein/Documents/GitHub/PoA/shp/Jing/journal.shp'
dir_data = 'C:/Users/Salomon Wollenstein/Documents/InverseVisTraffic/INRIX_data_gz' # Will take all of the csv files contained in folders and subfolders
files_ID = '_cdc_all_apr_2012'
dir_capacity_data = '../data/'

out_dir = '../results/' + files_ID + '/' 
filtered_data_dir =  out_dir + 'filtered_tmc_data' 	+ '/'

if os.path.isdir(out_dir) == False:
    os.mkdir(out_dir)
    os.mkdir(filtered_data_dir)
# Filtering by date range and tmc
dates_input = [#{'id':'Jan','start_date': '2015-01-01' , 'end_date':'2015-01-10'}, 
               #{'id':'Feb','start_date': '2015-02-01' , 'end_date':'2015-02-15'}] 
               {'id':'Apr','start_date': '2012-04-01' , 'end_date':'2012-04-2'}] 
               #{'id':'Aug','start_date': '2015-08-01' , 'end_date':'2015-08-10'}, 
               #{'id':'Nov','start_date': '2015-11-01' , 'end_date':'2015-11-10'}]
 
# Select of you want to analyze weekends or weekdays
days_of_week = 'weekdays'

dates = pd.DataFrame(dates_input)
percentile_free_flow = 85

# Time instances
time_instances_input = [{'id':'AM','start_time':'7:00', 'end_time':'9:00'}, 
                        {'id':'MD','start_time':'11:00', 'end_time':'13:00'}, 
                        {'id':'PM','start_time':'17:00', 'end_time':'19:00'}, 
                        {'id':'NT','start_time':'21:00', 'end_time':'23:00'}]

time_instances = pd.DataFrame(time_instances_input)

data_granularity = '1min'

#start_time = '9:00'
#end_time   = '11:00'

c_value_min = 0
confidence_score_min = 0


#OD Pairs, in this case, all combinations
od_pairs = []
for i in range(9)[1:]:
    for j in range(9)[1:]:
        if i != j:
            od_pairs.append([i, j])

od_pairs
number_of_routes_per_od = 3
theta = 0.8