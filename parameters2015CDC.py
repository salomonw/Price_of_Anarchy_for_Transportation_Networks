# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:50:08 2018

@author: Salomon Wollenstein
"""
import numpy as np
import pandas as pd
import os 

'Parameters'
dir_shpfile = 'G:/My Drive/Github/PoA/shp/Jing/journal.shp'
dir_data = 'G:/Team Drives/MPO 2015/INRIX_byQuarter/4-6' # Will take all of the csv files contained in folders and subfolders
files_ID = '_cdc_apr_2015'
dir_capacity_data = 'G:/Team Drives/MPO 2012/capacity data/'

out_dir = '../results/' + files_ID + '/' 
filtered_data_dir =  out_dir + 'filtered_tmc_data'  + '/'

if os.path.isdir(out_dir) == False:
    os.mkdir(out_dir)
    os.mkdir(filtered_data_dir)
# Filtering by date range and tmc
dates_input = [#{'id':'Jan','start_date': '2015-01-01' , 'end_date':'2015-01-10'}, 
               #{'id':'Feb','start_date': '2015-02-01' , 'end_date':'2015-02-15'}] 
               {'id':'Apr','start_date': '2015-04-01' , 'end_date':'2015-05-01'}] 
               #{'id':'Aug','start_date': '2015-08-01' , 'end_date':'2015-08-10'}, 
               #{'id':'Nov','start_date': '2015-11-01' , 'end_date':'2015-11-10'}]
 
''' ALWAYS WRITE ONE MORE DAY'''
    
    
# Select of you want to analyze weekends or weekdays
days_of_week = 'weekdays'

dates = pd.DataFrame(dates_input)
percentile_free_flow = 85

# Time instances
time_instances_input = [{'id':'AM','start_time':'7:00', 'end_time':'8:59'}, 
                        {'id':'MD','start_time':'11:00', 'end_time':'12:59'}, 
                        {'id':'PM','start_time':'17:00', 'end_time':'18:59'}, 
                        {'id':'NT','start_time':'21:00', 'end_time':'22:59'}]

instances_ID = ["AM", "MD", "PM", "NT"]
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
number_of_routes_per_od = 6
theta = 0.8
lower_bound_route = 0.02

average_over_time_p = 5 # averaging the samples to find OD Demand *min


#Inverse optimization
month_w = 'Apr'
month = 4
year = 2015
n_zones = 8

week_day_list_1 = [1 ,2 ,3 ,6 ,7 ,8 ,9 ,10] # train_idx = 1
week_day_list_2 = [13, 14, 15, 16, 17, 20, 21] # train_idx = 2
week_day_list_3 = [22, 23, 24, 27, 28 ,29, 30] # train_idx = 3
week_day_list = [1 ,2 ,3 ,6 ,7 ,8 ,9 ,10, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 27, 28 ,29, 30]

train_idx = range(1, 4)

#Machine learning parameters
deg_grid = range(4, 9)
c_grid = list(np.arange(.5, 3, .5))
lamb_grid = [10. ** m for m in range(-3, 4)]










