# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:00:53 2018

@author: Salomon Wollenstein
"""

' Packages '
import time
import os
import sys
import numpy as np
import pandas as pd
import os 
import networkx as nx
import multiprocessing as mp
#from dbfread import DBF    
import matplotlib.pyplot as plt
from gurobipy import *
import pickle
#import pysal as ps
import pandas as pd
import numpy as np
import datetime 

from utils import *


def flow_conservation_adjustment(G,y):
    # y is a dict with flows and TMCs
    y_0 = y.values()
    model = Model("Flow_conservation_adjustment")
    l = len(y)
    x = []
    # Define variables (adjusted flows)
    for i in range(l):
        x.append(model.addVar(name = str(y.keys()[i])))
    model.update()
    # Set objective ||x-y||2
    obj = 0
    for i in range(l):
        obj += (x[i] - y_0[i] ) * (x[i] - y_0[i])
    model.setObjective(obj)
    # Set constraints
    # non-negativity
    for i in range(l):
        model.addConstr( x[i] >= 0 )
    # conservation of flow
    for node in G.nodes():
        in_ = list(G.in_edges(nbunch = node,data=False))
        out_ = list(G.out_edges(nbunch = node,data=False)) 
        if len(in_)>0 and len(out_)>0:
            model.addConstr(quicksum(model.getVarByName(str(incoming_edge)) for incoming_edge in in_) == 
                    quicksum(model.getVarByName(str(outgoing_edge)) for outgoing_edge in out_))
    model.update() 
    model.setParam('OutputFlag', False)
    model.optimize()
    u = []
    res = {}
    for v in model.getVars():
        # print('%s %g' % (v.varName, v.x))
        #u.append(v.x)
        res[v.VarName] = v.x
     #print('Obj: %g' % obj.getValue())
    return res

# Retrive the TMCs data
#Filters TMC data for all csv files contained in folder the output is a set of files containing data 
def filter_TMC_mult_files(dir_data, files_ID, confidence_score_min, c_value_min, out_dir):
    df = pd.DataFrame()
    cnt = 0
    filtered_files_list = []
    dir_ = os.path.join(dir_data)
   # tmc_net_list = zload(out_dir + 'tmc_net_list' + files_ID + '.pkz')
    for root,dirs,files in os.walk(dir_):
        for file in files:
            df = pd.DataFrame()
            if file.endswith(".csv"):
                file_mem = os.stat(root + '/' +  file).st_size
                analized_mem = 0 
                iter_csv = pd.read_csv(root + '/' +  file, iterator=True, chunksize=200000)
                for chunk in iter_csv:
                    chunk['measurement_tstamp']=pd.to_datetime(chunk['measurement_tstamp'], format='%Y-%m-%d %H:%M:%S')
                    chunk = chunk.set_index('measurement_tstamp')
                    df2 = filter_tmc(chunk,tmc_net_list,confidence_score_min,c_value_min)   
                    df = df.append(df2)
                    chunk_mem = chunk.memory_usage(index=True).sum()
                    analized_mem += chunk_mem
                    cnt = cnt + 1
                    percentage = analized_mem/file_mem
                    filtered_files_list.append( out_dir + 'filtered_tmc_date_' + file[:-4]  +'.pkz' )
                    print(file + ' : ' + str(cnt))
                print('-----------------------------------------------------')
                pd.to_pickle(df, out_dir + 'filtered_tmc_' + file[:-4]  +'.pkz')
                del df
    

def filter_dates_and_free_flow_calc(dir_data, files_ID, out_dir, percentile_free_flow, dates_input):
# Create list of files to read ( this can be used when you have passed the first filter of data)
    dates = pd.DataFrame(dates_input)
    filtered_files_list = []
    dir_ = os.path.join(dir_data)
    for root,dirs,files in os.walk(dir_):
        for file in files:
            if file.endswith(".csv") or file.endswith(".pkz"):
                    filtered_files_list.append( dir_data + file[:-4]  +'.pkz' )
                    
    # Read filtered data and create a file with all the relevant data 
    df = pd.DataFrame()
    for filtered_file in filtered_files_list:
        df1 = pd.read_pickle(filtered_file)
        df = df.append(df1)
        del df1
    df.to_csv(out_dir + 'filtered_tmc' + files_ID + '.csv') # Save BIG data csv file with the filtered TMC data
    
    # Calculate percentiles 
    tmc_free_flow = df.groupby('tmc_code').agg(percentile(percentile_free_flow))['speed'] 
    tmc_free_flow.name= 'free_flow_speed'
    pd.to_pickle(tmc_free_flow, out_dir + 'free_flow_speed_ ' + files_ID + '.pkz')
    
    # filter specific dates of interst and export csv file , can be avoided once you are working with the same dates
    cnt = 0
    df = pd.DataFrame()
    for index, row in dates.iterrows():   
        iter_csv = pd.read_csv(out_dir + 'filtered_tmc' + files_ID + '.csv', iterator=True, chunksize=200000)
        for chunk in iter_csv:
            chunk['measurement_tstamp']=pd.to_datetime(chunk['measurement_tstamp'], format='%Y-%m-%d %H:%M:%S')
            chunk = chunk.set_index('measurement_tstamp')
            df2 = filter_dates(chunk, row['start_date'], row['end_date'])  
            df = df.append(df2)
            cnt = cnt + 1
            print(cnt)
        
        df.to_csv(out_dir + 'filtered_tmc_date' + files_ID +'.csv') 


def capacity_data(dir_capacity_data, files_ID, out_dir):  
    # tmc and roadinv lookup'
    tmc_net_list = zload(out_dir + 'tmc_net_list' + files_ID + '.pkz')
    lookup_tmc_roadinv = pd.read_excel(dir_capacity_data + 'roadinv_id_to_tmc_lookup.xlsx', index_col=None, na_values=['NA'], parse_cols = "A,D")
    lookup_tmc_roadinv = lookup_tmc_roadinv.set_index('ROADINV_ID')
    lookup_tmc_roadinv = lookup_tmc_roadinv[lookup_tmc_roadinv['TMC'].isin(tmc_net_list)]
    
    # Load capacity file'
    
    cap_data = pd.read_excel(dir_capacity_data + 'capacity_attribute_table.xlsx', index_col=None, na_values=['NA'], parse_cols = "B,H,J,L,N,P,R,T,V,X,Z")
    cap_data.rename(columns={'SCEN_00_AB': "AB_AMLANE", 'SCEN_00_A1': "AB_MDLANE", 'SCEN_00_A2': 'AB_MDLANE', 'SCEN_00_A3': 'AB_PMLANE' }, inplace=True)
    
    cap_data = cap_data.set_index('ROADINVENT')
    cap_data.index = cap_data.index.fillna(0).astype(np.int64)
        
     # take the period capacity factor into consideration
    cap_data.AB_AMCAPAC = (1.0/2.5) * cap_data.AB_AMCAPAC 
    cap_data.AB_MDCAPAC = (1.0/4.75) * cap_data.AB_MDCAPAC
    cap_data.AB_PMCAPAC = (1.0/2.5) * cap_data.AB_PMCAPAC
    cap_data.AB_NTCAPAC = (1.0/7.0) * cap_data.AB_NTCAPAC
    
    result = cap_data
    result  = result.join(lookup_tmc_roadinv, how='inner')
    result = result.set_index('TMC')


    pd.to_pickle(lookup_tmc_roadinv,  out_dir + 'lookup_tmc_roadinv' + files_ID + '.pkz')
    pd.to_pickle(result, out_dir + 'capacity_data_' + files_ID + '.pkz')
    pd.to_pickle(cap_data, out_dir + 'cap_data' + files_ID + '.pkz')


    
def calculate_ref_speed_tables(out_dir, files_ID):
    tmc_net_list = zload(out_dir + 'tmc_net_list' + files_ID + '.pkz')
    df = pd.read_csv(out_dir + 'filtered_tmc_date' + files_ID + '.csv')
    df['measurement_tstamp'] = pd.to_datetime(df.measurement_tstamp)
    df = df.set_index('measurement_tstamp')
    df['dayWeek'] = df.index.dayofweek
    df['min'] = (df.index.hour) * 60 +  df.index.minute
    df = df.reset_index()
    ref_speed_collection = {}
    for tmc in tmc_net_list:
        df2 = df[df['tmc_code']==tmc]
        df2 = pd.pivot_table(df2, index='min', columns=['dayWeek'], values = 'speed' , aggfunc = np.median) 
        u = pd.DataFrame(index=range(0,1439),columns=[0,1,2,3,4,5,6])    

        if df2.empty == False:
            u.update(df2)

        for col in u:
            u[col] = pd.to_numeric(u[col], errors='coerce')

        ref_speed_collection[tmc] = u.interpolate()

    zdump(ref_speed_collection, out_dir + 'ref_speed_collection' + files_ID + '.pkz')


def filter_time_instances(out_dir, files_ID, time_instances, data_granularity):
    # filtering the time instances that the user is interested and exporting csv files for each of these
    tmc_net_list = zload(out_dir + 'tmc_net_list' + files_ID + '.pkz')
    capacity_df = pd.read_pickle(out_dir + 'capacity_data_' + files_ID + '.pkz')
    tmc_free_flow = pd.read_pickle(out_dir + 'free_flow_speed_ ' + files_ID + '.pkz')
    lookup_tmc_roadinv = pd.read_pickle(out_dir + 'lookup_tmc_roadinv' + files_ID + '.pkz')
    cap_data = pd.read_pickle(out_dir + 'cap_data' + files_ID + '.pkz')
    ref_speed_collection = zload(out_dir + 'ref_speed_collection' + files_ID + '.pkz')
    
    for index, row in time_instances.iterrows():
        df = pd.DataFrame()
        cnt = 0
        iter_csv = pd.read_csv(out_dir + 'filtered_tmc_date' + files_ID + '.csv', iterator=True, chunksize=200000)
        for chunk in iter_csv:
            df2 = filter_time(chunk, row['start_time'], row['end_time'])  
            df = df.append(df2)
            cnt = cnt + 1
            print(cnt)        
        del df2

        # creating a table with the characteristics of TMCs in instance AM,MD,PM,NT    
        tmc_instance_stats = df.groupby('tmc_code').agg(np.mean)
        result2 = capacity_df.join(tmc_instance_stats, how='inner')
        result2 = pd.merge(lookup_tmc_roadinv, cap_data, right_index=True, left_index=True)
        result2 = result2.set_index('TMC')
        tmc_instance_char = tmc_instance_stats.join(result2, how='outer')
        tmc_instance_char = tmc_instance_char[~tmc_instance_char.index.duplicated(keep='first')]
        tmc_instance_char = tmc_instance_char.join(tmc_free_flow, how='outer')
        tmc_instance_char = tmc_instance_char.to_dict()
        pd.to_pickle(df, out_dir + 'filtered_tmc_date_time' + files_ID + '_' + row['id'] +'.pkz')   
        df.reset_index(level=0, inplace=True)
        
        # calculating dataflows
        df['measurement_tstamp'] = pd.to_datetime(df.measurement_tstamp)
        df = (df.set_index('measurement_tstamp').groupby('tmc_code').resample(data_granularity).mean().reset_index())
        df = filter_time(df,row['start_time'], row['end_time'])
        df['dayWeek'] = df.index.dayofweek
        df['min'] = df.index.hour * 60 +  df.index.minute
        df['idx'] = range(1, len(df) + 1)
        df = df.reset_index()
        a=[]
        for idx, row2 in df.iterrows():
            capacity = tmc_instance_char['AB_'+ time_instances.id[index] +'CAPAC'][row2['tmc_code']]
            free_flow_sp = tmc_instance_char['free_flow_speed'][row2['tmc_code']]
            speed = row2['speed']
            if np.isnan(speed)  == True:
                speed = ref_speed_collection[row2['tmc_code']].iloc[row2['min']-1, row2['dayWeek']-1]
                df.set_value(idx,'speed', speed)
            x_flow = greenshield(min(speed,free_flow_sp) , capacity , free_flow_sp)
            a.append([row2['idx'], x_flow])
        
        a = pd.DataFrame(a)
        a = a.rename(index=str, columns={0: "idx", 1 : "xflow"})
        
        df = df.join(a.set_index('idx'), on='idx')

        # if there is still missing values, then interpolate
        #if df['speed'].isnull().values.any():
        #    df['speed'] = df['speed'].interpolate('index')
        #    for row3 in (df.loc[pd.isnull(df['speed']), 'speed']).index:
        #        speed = df.iloc[row3,'speed']
        #        capacity = tmc_instance_char['AB_'+ time_instances.id[index] +'CAPAC'][df.iloc[row3,'tmc_code']]
        #        free_flow_sp = tmc_instance_char['free_flow_speed'][df.iloc[row3,'tmc_code']]
        #        x_flow = greenshield(min(speed,free_flow_sp) , capacity , free_flow_sp)
        #        a.append([df.iloc[row3,'idx'], x_flow])



        del a, chunk, 
    
        pd.to_pickle(df, out_dir + 'filtered_tmc_date_time_flow' + files_ID + '_' + row['id'] +'.pkz')   
        pd.to_pickle(result2, out_dir + 'result_2' + files_ID + '_' + row['id'] +'.pkz') #!!!!!!!!!!!!!!! RENAME !!!!!!!!!!!
        
def calculate_data_flows(out_dir, files_ID, time_instances, days_of_week):
    # Filtering by weekdays/weekends
    G = zload(out_dir + 'G' + files_ID + '.pkz')
    link_tmc_dict = zload(out_dir + 'link_tmc_dict' + files_ID + '.pkz')
    free_flow_speed = pd.read_pickle(out_dir + 'free_flow_speed_ ' + files_ID + '.pkz')
    tmc_att = zload(out_dir + 'tmc_att' + files_ID + '.pkz')
    tmc_att = tmc_att.set_index('TMC')
    
    G_ = {}
    for instance in list(time_instances['id']):
        link_ = list()
        link_flow = {}
        linkFlow = pd.DataFrame()
        df = pd.read_pickle(out_dir + 'filtered_tmc_date_time_flow' + files_ID + '_' + instance +'.pkz')
        result2 = pd.read_pickle(out_dir + 'result_2' + files_ID + '_' + instance + '.pkz') #!!!!!!!!!!!!!!! RENAME !!!!!!!!!!!
        result2 = result2[~result2.index.duplicated(keep='first')]
        df = df.join(tmc_att['Shape_Leng'], on = 'tmc_code', how = 'inner')
        if days_of_week == 'weekdays':
            df = df[df['dayWeek']>0]
            df = df[df['dayWeek']<7]
        
        # TMC to link aggregation  
        l_length = {}
        l_avgSpeed = {}
        tmc_edge_df = pd.DataFrame(link_tmc_dict.items(),columns = ['TMC','link'])
        for link in list(tmc_edge_df.link.unique()):
            l_xflows = pd.DataFrame()
            tmc_list = tmc_edge_df[tmc_edge_df['link']==link]['TMC']
            df2 = df[df['tmc_code'].isin(tmc_list)]
            #df2['prod'] = df2['xflow'] * df2['LENGTH']
            #grouped = df2.groupby('measurement_tstamp').sum()
            #l_xflows ['flow'] = grouped['prod'] / grouped['LENGTH'] 
            df2['Shape_Leng'] =  0.000621371 * df2['Shape_Leng']  #since the speed (miles/hr) and len (meters --> miles)
            df2['prod'] = ( df2['xflow'] * df2['Shape_Leng']) / df2['speed']
            df2['travelTime'] = df2['Shape_Leng'] / df2['speed']
            grouped = df2.groupby('measurement_tstamp').sum()
            
            #sum_avg_speed_free_flow = sum(df2.groupby('measurement_tstamp').mean()['speed'])
            #free_flow_speed_link = free_flow_speed[free_flow_speed.index.isin(tmc_list)]
            l_xflows['flow'] = grouped['prod']/(grouped['Shape_Leng']/ grouped['travelTime'] )
            
            if l_xflows.isnull().values.any() == True:
                time.sleep()
            summary = df2.groupby('measurement_tstamp').mean()
            tmc_length = df2.groupby('tmc_code').mean()['Shape_Leng']
            l_length[link] =  sum(tmc_length)
            tmc_avgSpeed = df2.groupby('tmc_code').mean()['speed']
            l_avgSpeed[link] = sum(tmc_avgSpeed*tmc_length)/sum(tmc_length)
            l_avgTravelTime = df2.groupby('measurement_tstamp').sum()['travel_time'] ### travel_time_minutes -> for 2015 , travel_time -> for 2012 
            link_flow[link] = l_xflows
            linkFlow = linkFlow.append(l_xflows)
            link_.extend([link]*len(l_xflows))
        
        linkFlow['link'] = link_
        linkFlow = linkFlow.reset_index()
        unique_t = linkFlow['measurement_tstamp'].unique()

        G_[instance] = G
        G_[instance] = nx.set_edge_attributes(G , name = 'length', values= l_length )
        G_[instance] = nx.set_edge_attributes(G , name = 'avgSpeed', values= l_avgSpeed)

        flow_after_conservation={}
        for idx in list(unique_t):
            ins = linkFlow[linkFlow['measurement_tstamp']==idx]
            ins = ins[['link','flow']].set_index('link').to_dict()['flow']
            if len(ins) == len(link_flow): #if there is no data of one llnk for an instance, then delete it
                flow_after_conservation[idx] = flow_conservation_adjustment(G,ins)
            
        pd.to_pickle(flow_after_conservation, out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
    return G_
    zdump(G_, out_dir + 'G_' + files_ID +  +'.pkz' )
    
  
#### ------------ OD Estimation ------------- :
    
    
def createPathLinkIncidenceMatrix(G, link_length, link_freeFlowTravelTime):
    nodes = list(G.nodes())
    neighbors={}
    for node in nodes:
        neighbors_dict = (G.neighbors(node))
        
    # read link length dictionary
    
    # read link free Flow Travel time
    
    
    # compute the lenght of a route
    
    def routeTime(route):
        link_list = []
        node_list = []
        for i in route.split('->'):
            node_list.append(int(i))
        for i in range(len(node_list))[:-1]:
            link_list.append('%d->%d' %(node_list[i], node_list[i+1]))
        time_of_route = sum([link_freeFlowTravelTime_dict[str(link_label_dict_[link])] for link in link_list])
        return time_of_route


    # export a Path link incidence Matrix
        #Maybe use simple paths proposed by networkx
    
