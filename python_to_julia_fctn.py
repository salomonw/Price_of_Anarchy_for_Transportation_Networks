# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 23:39:57 2018

@author: Salomon Wollenstein
"""
from utils import *
import json
import numpy as np
import os 

def parse_data_for_Julia(out_dir, files_ID, time_instances):
    import collections
    G_ = zload(out_dir + 'G_' + files_ID + '.pkz' )
    free_flow_link = zload(out_dir + 'free_flow_link' + files_ID + '.pkz' )
    G = G_[time_instances['id'][0]]
    link_avg_flow = zload(out_dir + 'link_avg_day_flow' + files_ID + '.pkz')
    capacity_link = zload(out_dir + 'capacity_link' + files_ID + '.pkz')
    
    edges = list(G.edges())
    
    link_edge_dict = {}
    link_min_dict = {}
    link_length_dict = {}
    idx = 0
    
    # Read flows
    flows_after = {}
    flows_before = {}
    speed_before = {}
    density = {}
    flows_after_k = {}
    speed_before_k = {}
    density_k  = {}
    for instance in time_instances['id']:   
        flows_before_0 = pd.read_pickle(out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance +'.pkz')
        flows_before[instance] = collections.OrderedDict(sorted(flows_before_0.items()))
        flows_after_0 =  pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
        flows_after[instance] = collections.OrderedDict(sorted(flows_after_0.items()))
        
        flows_after_k1 = list(flows_after[instance].keys())
        flows_after_k[instance] = pd.DatetimeIndex(flows_after_k1)
        
        speed_before_0 = pd.read_pickle(out_dir + 'speed_links' + files_ID + '_' + instance +'.pkz')
        speed_before[instance] = collections.OrderedDict(sorted(speed_before_0.items()))
        
        speed_before_k1 = list(speed_before[instance].keys())
        speed_before_k[instance] = pd.DatetimeIndex(speed_before_k1)
        
        density_0 = pd.read_pickle(out_dir + 'density_links' + files_ID + '_' + instance +'.pkz')
        density[instance] = collections.OrderedDict(sorted(density_0.items()))

        density_k1 = list(density[instance].keys())
        density_k[instance] = pd.DatetimeIndex(density_k1)
        
        uniq_days = flows_after_k[instance].normalize().unique()
        
    for edge in edges:        
        from_ = edge[0]
        to_ = edge[1]  
        edge_len = G.get_edge_data(from_,to_)['length']
        free_flow_speed = free_flow_link[edge]
        

        
        for i in uniq_days:
            day = np.asscalar(pd.DatetimeIndex([i]).day)
            month = np.asscalar(pd.DatetimeIndex([i]).month)
            year = np.asscalar(pd.DatetimeIndex([i]).year)  
            key = {}

            key['day'] = day
            key['month'] = month
            key['year'] = year            
            key['avg_speed'] = G.get_edge_data(from_,to_)['avgSpeed']
            key['edge_len'] = edge_len
            key['free_flow_speed'] = free_flow_speed
            key['free_flow_time'] = edge_len/free_flow_speed
            key['init_node'] = from_
            key['term_node'] = to_
            
            for instance in time_instances['id']:

                flows_after_k2 = flows_after_k[instance]
                speed_before_k2 = speed_before_k[instance]
                density_k2 = density_k[instance]
                
                keys_ = 'link_' + str(edge) + '_' + str(year) + '_' + str(month) + '_' + str(day)
                from_date = datetime.datetime(year,month,day,0,0,0)
                to_date = from_date + datetime.timedelta(days = 1)
                
                idx2 = (flows_after_k2 > from_date) & (flows_after_k2 <= to_date)
                flow_days = flows_after_k2[idx2]
                speed_days = speed_before_k2[idx2]
                density_days = density_k2[idx2]
                
                flow_days_dict = { your_key: flows_after[instance][your_key] for your_key in flow_days.values }
                flow_days_dict = collections.OrderedDict(sorted(flow_days_dict.items()))
                
                speed_days_dict = { your_key: speed_before[instance][your_key] for your_key in speed_days.values }
                speed_days_dict = collections.OrderedDict(sorted(speed_days_dict.items()))
                
                density_days_dict = { your_key: density[instance][your_key] for your_key in density_days.values }
                density_days_dict = collections.OrderedDict(sorted(density_days_dict.items()))
                
                flow_avg = link_avg_flow[str(edge) + '_' + instance]
                
                idx3 = (flow_avg.index > from_date) & (flow_avg.index <= to_date)
                flow_avg = flow_avg[idx3]
                key['avg_flow_' + instance] = flow_avg['flow'][0]
                
                key['capac_'+ instance] = capacity_link[str(edge) + '_' + instance]
                
                v = []
                u = []
                x = []
                for j in flow_days_dict.keys():
                    v.append(flow_days_dict[j][str(edge)])
                    u.append(speed_days_dict[j][edge])
                    x.append(density_days_dict[j][str(edge)])
                    #print(flow_days_dict[j][str(edge)])
                    
                key['flow_'+ instance] = v
                key['speed_'+ instance] = u
                key['denisty_'+ instance] = x
                
            link_min_dict[keys_] = key
            


        idx += 1
        link_edge_dict[idx] = edge
        link_length_dict[idx] = edge_len
        print('link: ' + str(edge) + ' has been processed ' + str(idx)  + ' !')
    
    with open(out_dir + 'link_edge_dict' + files_ID  + '.json', 'w') as fp:
            json.dump(link_edge_dict, fp)
    
    with open(out_dir + 'link_min_dict' + files_ID  + '.json', 'w') as fp:
            json.dump(link_min_dict, fp)

    with open(out_dir + 'link_length_dict' + '.json', 'w') as fp:
            json.dump(link_length_dict, fp)
    
    zdump(link_edge_dict, out_dir + 'link_edge_dict' + files_ID + '.pkz')
    zdump(link_min_dict, out_dir + 'link_min_dict' + files_ID + '.pkz')
    zdump(link_length_dict, out_dir + 'link_length_dict'  + '.pkz')





def create_testing_set(train_list_1, train_list_2, train_list_3, year, month, time_instances, month_w, out_dir, files_ID):
    import gurobipy
    from numpy.linalg import inv
    from numpy import linalg as LA
  
    # Create testing sets
    for instance in time_instances['id']:
        with open(out_dir + 'link_min_dict'+ files_ID + '.json', 'r') as json_file:
            link_day_minute_Apr_dict_JSON = json.load(json_file)
            
        link_ = zload(out_dir + 'link_edge_dict' + files_ID + '.pkz')
        link_flow_testing_set_Apr_AM_1 = []
        for link_idx in link_.keys():
            for day in train_list_1:
                key = 'link_' + str(link_[link_idx]) + '_' + str(year) + '_' + str(month) + '_' + str(day)
                link_flow_testing_set_Apr_AM_1.append(link_day_minute_Apr_dict_JSON[key] ['avg_flow_'+ instance])
        
        link_flow_testing_set_Apr_AM_2 = []
        for link_idx in link_.keys():
            for day in train_list_2:
                key = 'link_' + str(link_[link_idx]) + '_' + str(year) + '_' + str(month) + '_' + str(day)
                link_flow_testing_set_Apr_AM_2.append(link_day_minute_Apr_dict_JSON[key] ['avg_flow_'+ instance])
        
        link_flow_testing_set_Apr_AM_3 = []
        for link_idx in link_.keys():
            for day in train_list_3: 
                key = 'link_' + str(link_[link_idx]) + '_' + str(year) + '_' + str(month) + '_' + str(day)
                link_flow_testing_set_Apr_AM_3.append(link_day_minute_Apr_dict_JSON[key] ['avg_flow_'+ instance])
                
        n_links = len(link_.keys())
        testing_set_1 = np.matrix(link_flow_testing_set_Apr_AM_1)
        testing_set_1 = np.matrix.reshape(testing_set_1, n_links, len(train_list_1))
        testing_set_1 = np.nan_to_num(testing_set_1)
        y = np.array(np.transpose(testing_set_1))
        y = y[np.all(y != 0, axis=1)]
        testing_set_1 = np.transpose(y)
        testing_set_1 = np.matrix(testing_set_1)
        
        testing_set_2 = np.matrix(link_flow_testing_set_Apr_AM_2)
        testing_set_2 = np.matrix.reshape(testing_set_2, n_links, len(train_list_2))
        testing_set_2 = np.nan_to_num(testing_set_2)
        y = np.array(np.transpose(testing_set_2))
        y = y[np.all(y != 0, axis=1)]
        testing_set_2 = np.transpose(y)
        testing_set_2 = np.matrix(testing_set_2)
        
        testing_set_3 = np.matrix(link_flow_testing_set_Apr_AM_3)
        testing_set_3 = np.matrix.reshape(testing_set_3, n_links, len(train_list_3))
        testing_set_3 = np.nan_to_num(testing_set_3)
        y = np.array(np.transpose(testing_set_3))
        y = y[np.all(y != 0, axis=1)]
        testing_set_3 = np.transpose(y)
        testing_set_3 = np.matrix(testing_set_3)
        
        #np.size(testing_set_2, 0), np.size(testing_set_3, 1)
        #testing_set_3[:,:1]
        
        zdump([testing_set_1, testing_set_2, testing_set_3], out_dir + 'testing_sets_'+ month_w  + '_' + instance + '.pkz')
        
    
def calculate_testing_errors(out_dir, files_ID, month_w, instance, deg_grid, c_grid, lamb_grid):
    from numpy import arange
    # load testing sets (link flows)
    testing_set_1, testing_set_2, testing_set_3 = zload( out_dir + 'testing_sets_' + month_w + '_' + instance + '.pkz')
    
    testing_sets = {}
    testing_sets[1] = testing_set_1
    testing_sets[2] = testing_set_2
    testing_sets[3] = testing_set_3
    
    # load link flow data (solution of the forward problem corresponding to trained costs)
    with open( out_dir + 'uni-class_traffic_assignment_MSA_flows_Apr_AM.json', 'r') as json_file:
        xl = json.load(json_file)
    
    # create a dictionary to store testing errors
    testing_errors_dict = {}
    train_idx = range(1, 4)
    
    for deg in deg_grid:
        for c in c_grid:
            for lam in lamb_grid:
                for idx in train_idx:
                    key_ = "'(" + str(deg) + ',' + str(c) + ',' + str(lam) + ',' + str(idx) + ")'"
                    #if lam == 1e-5:
                    #    key_ = "'(" + str(deg) + ',' + str(c) + ',' + '1.0e-5' + ',' + str(idx) + ")'"
                    key_ = key_[1:-1]
                    testing_errors_dict[key_] = np.mean([LA.norm(np.array(xl[key_]) - \
                                                                 np.array(testing_sets[idx])[:, j]) \
                                                         for j in range(np.size(testing_sets[idx], 1))])
    
    testing_mean_errors_dict = {}
    train_idx = range(1, 4)
    
    for deg in deg_grid:
        for c in c_grid:
            for lam in lamb_grid:
                key_ = {}
                for idx in train_idx:
                    key_[idx] = "'(" + str(deg) + ',' + str(c) + ',' + str(lam) + ',' + str(idx) + ")'"
                    #if lam == 1e-5:
                    #    key_[idx] = "'(" + str(deg) + ',' + str(c) + ',' + '1.0e-5' + ',' + str(idx) + ")'"
                    key_[idx] = key_[idx][1:-1]
                key__ = "'(" + str(deg) + ',' + str(c) + ',' + str(lam) + ")'"
                #if lam == 1e-5:
                #    key__ = "'(" + str(deg) + ',' + str(c) + ',' + '1.0e-5' + ")'"
                key__ = key__[1:-1]
                
                testing_mean_errors_dict[key__] = np.mean([testing_errors_dict[key_[idx]] for idx in train_idx])


    testing_mean_errors_switch_dict = {}
    for _key_ in testing_mean_errors_dict.keys():
        testing_mean_errors_switch_dict[testing_mean_errors_dict[_key_]] = _key_
        

    #best_key = '(8,0.5,10000.0,1)'
    
    # Writing JSON data
    with open(out_dir + 'cross_validation_best_key_' + month_w + '_'+ instance + '.json', 'w') as json_file:
        json.dump(best_key, json_file)
        
        
def create_East_Massachusetts_trips(out_dir, files_ID, month_w, time_instances, n_zones, week_day_list):
    week_day_list.append("full")

    if not os.path.exists(out_dir + 'data_traffic_assignment_uni-class'):
            os.mkdir(out_dir + 'data_traffic_assignment_uni-class')
            
    for instance in time_instances['id']: 
        #list_of_lists = []
        for day in week_day_list:
            list_of_lists = []
            with open(out_dir  + 'OD_demands/OD_demand_matrix_' + month_w + '_' + str(day) + '_weekday_' + instance + files_ID +'.txt', 'r') as the_file:
                idx = 0
                for line in the_file:
                    inner_list = [elt.strip() for elt in line.split(',')]
                    list_of_lists.append(inner_list)
            
            zero_value = 0.0
            with open(out_dir + 'data_traffic_assignment_uni-class/'+ files_ID + '_trips_' + month_w + '_' + str(day) + '_' + instance +".txt", "w") as text_file:
                text_file.write("<NUMBER OF ZONES> "+ str(n_zones) +"\n")
                text_file.write("<TOTAL OD FLOW> 0.0\n")
                text_file.write("<END OF METADATA>\n\n\n")
                
                for n in range(n_zones):
                    text_file.write("Origin  %d  \n" %(n+1))
                    text_file.write("%d :      0.0;    " %(n+1))
                    for idx in range(n*(n_zones-1), (n+1)*(n_zones-1)):
                        text_file.write("%d :      %f;    " \
                                        %(int(list_of_lists[idx][1]), float(list_of_lists[idx][2])))
                        if idx % 3 == 0:
                            text_file.write("\n")
                    text_file.write("\n\n")
        
    
def create_East_Massachusetts_net(out_dir, files_ID, month_w, month, year, time_instances, n_zones, week_day_list):
    
    week_day_list.append("full")

    with open(out_dir + 'link_min_dict'+ files_ID + '.json', 'r') as json_file:
        link_day_minute_Apr_dict_JSON_ = json.load(json_file)
    zero_value = 0.0
    G_ = zload(out_dir + 'G_' + files_ID + '.pkz' )
    feas_day = link_day_minute_Apr_dict_JSON_[link_day_minute_Apr_dict_JSON_.keys()[0]]['day']
    
    for instance in time_instances['id']: 
        G = G_[instance]
        edges = list(G.edges())
        n_links = len(edges)
        
        for day in week_day_list:
            with open(out_dir + 'data_traffic_assignment_uni-class/'+ files_ID + '_net_' + month_w + '_' + str(day) + '_' +  instance +".txt", "w") as text_file:
                text_file.write("<NUMBER OF ZONES> "+ str(n_zones) +"\n")
                text_file.write("<NUMBER OF ZONES> "+ str(n_zones) +"\n")
                text_file.write("<FIRST THRU NODE> 1\n")
                text_file.write("<NUMBER OF LINKS> "+ str(n_links) +"\n")
                text_file.write("<NUMBER OF ZONES> "+ str(n_zones) +"\n")
                text_file.write("<END OF METADATA>\n\n\n")
                text_file.write("~  Init node  Term node  Capacity  Length  Free Flow Time  B  Power  Speed limit  Toll  Type  ;\n")
                for idx in edges:
                     text_file.write("  %d  %d  %f  %f  %f  %f  %f  %f  %f  %f  ;\n" \
                                     %(link_day_minute_Apr_dict_JSON_['link_' + str(idx) + '_' + str(year) + '_' + str(month) + '_' + str(feas_day)]['init_node'], \
                                       link_day_minute_Apr_dict_JSON_['link_' + str(idx) + '_' + str(year) + '_' + str(month) + '_' + str(feas_day)]['term_node'], \
                                       link_day_minute_Apr_dict_JSON_['link_' + str(idx) + '_' + str(year) + '_' + str(month) + '_' + str(feas_day)]['capac_' + instance], \
                                       zero_value, \
                                       link_day_minute_Apr_dict_JSON_['link_' + str(idx) + '_' + str(year) + '_' + str(month) + '_' + str(feas_day)]['free_flow_time'], \
                                       zero_value, zero_value, zero_value, zero_value, zero_value))
            
            
            
    #def calculating_testing_errors(out_dir, files_ID, month_w, month, year, time_instances, n_zones):
        