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
from math import exp
from utils import *

from numpy import linalg as LA

def adj_PSD(Sigma):
    # Ensure Sigma to be symmetric
    Sigma = (1.0 / 2) * (Sigma + np.transpose(Sigma))

    # Ensure Sigma to be positive semi-definite
    D, V = LA.eig(Sigma)
    D = np.diag(D)
    Q, R = LA.qr(V)
    for i in range(0, np.size(Sigma,0)):
        if D[i, i] < 1e-5:
            D[i, i] = 1e-5
    Sigma = np.dot(np.dot(Q, D), LA.inv(Q))
    return Sigma


def add_const_diag(X, lam):
    X = np.matrix(X)
    len_x = len(X)
    Y = X + np.eye(len_x) * lam 
    return Y


def isPSD(A, tol=1e-8):
  E,V = np.linalg.eigh(A)
  return np.all(E > -tol)

def samp_cov(x):
    """
    x: sample matrix, each column is a link flow vector sample
    K: number of samples
    S: sample covariance matrix
    ----------------
    return: S
    ----------------
    """
    x = np.matrix(x)
    K = np.size(x, 1)
    x_mean = sum(x[:,k] for k in range(K)) / K
    S = sum(np.dot(x[:,k] - x_mean, np.transpose(x[:,k] - x_mean)) for k in range(K)) / (K - 1)
   # S = adj_PSD(S)
    return S


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
        #u.append(v.x)S
        res[v.VarName] = v.x
     #print('Obj: %g' % obj.getValue())
    return res

# Retrive the TMCs data
#Filters TMC data for all csv files contained in folder the output is a set of files containing data 
def filter_TMC_mult_files(dir_data, files_ID, confidence_score_min, c_value_min, out_dir, filtered_data_dir):
    df = pd.DataFrame()
    cnt = 0
    filtered_files_list = []
    dir_ = os.path.join(dir_data)
    tmc_net_list = zload(out_dir + 'tmc_net_list' + files_ID + '.pkz')
    for root,dirs,files in os.walk(dir_):
        for file in files:
            df = pd.DataFrame()
            if file.endswith(".csv"):
                file_mem = os.stat(root + '/' +  file).st_size
                analized_mem = 0 
                iter_csv = pd.read_csv(root + '/' +  file, iterator=True, chunksize=200000)
                for chunk in iter_csv:
                    #chunk = chunk.reset_index()
                    #chunk.columns = ['tmc_code', '']
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
                pd.to_pickle(df, filtered_data_dir + 'filtered_tmc_' + file[:-4]  +'.pkz')
                del df
    

def filter_dates_and_free_flow_calc(dir_data, files_ID, out_dir, percentile_free_flow, dates_input):
# Create list of files to read ( this can be used when you have passed the first filter of data)
    dates = pd.DataFrame(dates_input)
    filtered_files_list = []
    dir_ = os.path.join(dir_data)
    for root,dirs,files in os.walk(dir_):
        for file in files:
            if file.endswith(".pkz"):
            #if file.endswith(".csv") or file.endswith(".pkz"):
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
    tmc_free_flow = pd.read_pickle(out_dir + 'free_flow_speed_ ' + files_ID + '.pkz') ##_________________________________
    lookup_tmc_roadinv = pd.read_pickle(out_dir + 'lookup_tmc_roadinv' + files_ID + '.pkz')
    cap_data = pd.read_pickle(out_dir + 'cap_data' + files_ID + '.pkz')
    ref_speed_collection = zload(out_dir + 'ref_speed_collection' + files_ID + '.pkz')
    
    for index, row in time_instances.iterrows():
        df = pd.DataFrame()
        cnt = 0
        cnt_1 = 0
        iter_csv = pd.read_csv(out_dir + 'filtered_tmc_date' + files_ID + '.csv', iterator=True, chunksize=200000)
        for chunk in iter_csv:
            df2 = filter_time(chunk, row['start_time'], row['end_time'])  
            df = df.append(df2)
            cnt = cnt + 1
            print(cnt)        
        del df2
        
        print('file readed!, calculating dataflows for instance: ' + row[1] + '...')
        
        df = df.reset_index()
        df['tmc_date'] = df['tmc_code']  + '_' + row[1] + '_'  + df['measurement_tstamp'].dt.date.map(str)
        avg_speed = df.groupby('tmc_date').mean()
        avg_speed['avg_speed_day'] = avg_speed['speed']
        df = df.set_index('tmc_date')
        df = df.join(avg_speed['avg_speed_day'], on = 'tmc_date', how = 'inner')
        df = df.set_index('measurement_tstamp')
        
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
        avg = []
        sp = []
        for idx, row2 in df.iterrows():
            capacity = tmc_instance_char['AB_'+ time_instances.id[index] +'CAPAC'][row2['tmc_code']]
            length = tmc_instance_char['LENGTH'][row2['tmc_code']]
            num_lanes = tmc_instance_char['AB_AMLANE'][row2['tmc_code']]
            free_flow_sp = tmc_instance_char['free_flow_speed'][row2['tmc_code']]
            speed = row2['speed']
            avg_speed = row2['avg_speed_day']
            if np.isnan(speed)  == True:
                try:
                    speed = ref_speed_collection[row2['tmc_code']].iloc[row2['min']-1, row2['dayWeek']-1]
                except:
                    speed = 0.0000001 
                    cnt_1 += 1
                df.set_value(idx,'speed', speed)
            x_flow = greenshield(min(speed, free_flow_sp) , capacity , free_flow_sp)
            avg_flow = greenshield(min(avg_speed, free_flow_sp) , capacity , free_flow_sp) # check avg_speed for 2015 dataset
            a.append([row2['idx'], x_flow])
            avg.append([row2['idx'], avg_flow])
            sp.append([row2['idx'], speed])
            
        a = pd.DataFrame(a)
        a = a.rename(index=str, columns={0: "idx", 1 : "xflow"})
        
        avg = pd.DataFrame(avg)
        avg = avg.rename(index=str, columns={0: "idx", 1 : "avg_flow"})
        
        sp = pd.DataFrame(sp)
        sp= sp.rename(index=str, columns={0: "idx", 1 : "speed2"})
        
        df = df.join(a.set_index('idx'), on='idx')
        df = df.join(avg.set_index('idx'), on='idx')
        df = df.join(sp.set_index('idx'), on='idx')
        
        del a, chunk, avg, sp
    
        pd.to_pickle(df, out_dir + 'filtered_tmc_date_time_flow' + files_ID + '_' + row['id'] +'.pkz')   
        pd.to_pickle(result2, out_dir + 'result_2' + files_ID + '_' + row['id'] +'.pkz') #!!!!!!!!!!!!!!! RENAME !!!!!!!!!!!
        print('there are: ' + str(cnt_1) + 'missing values')
        
def calculate_data_flows(out_dir, files_ID, time_instances, days_of_week):
    import json
    # Filtering by weekdays/weekends
    G = zload(out_dir + 'G' + files_ID + '.pkz')
    link_tmc_dict = zload(out_dir + 'link_tmc_dict' + files_ID + '.pkz')
    free_flow_speed = pd.read_pickle(out_dir + 'free_flow_speed_ ' + files_ID + '.pkz')
    tmc_att = zload(out_dir + 'tmc_att' + files_ID + '.pkz')
    tmc_att = tmc_att.set_index('TMC')
    free_flow_link = {}
    G_ = {}
    link_avg_flow = {}
    capacity_link = {}
    
    for instance in list(time_instances['id']):
        link_ = list()
        link_flow = {}
        linkFlow = pd.DataFrame()
        speed_flow = {}
        linkSpeed= pd.DataFrame()
        link_density = {}
        linkDensity = pd.DataFrame()
        
        df = pd.read_pickle(out_dir + 'filtered_tmc_date_time_flow' + files_ID + '_' + instance +'.pkz')
        result2 = pd.read_pickle(out_dir + 'result_2' + files_ID + '_' + instance + '.pkz') #!!!!!!!!!!!!!!! RENAME !!!!!!!!!!!
        result2 = result2[~result2.index.duplicated(keep='first')]
        df = df.join(tmc_att['Shape_Leng'], on = 'tmc_code', how = 'inner')
        df['tmc_date'] = df['tmc_code']  + '_' + instance + '_'  + df['measurement_tstamp'].dt.date.map(str)
        
        if days_of_week == 'weekdays':
           # df = df[df['dayWeek']>0]
            df = df[df['dayWeek']<5]
        
        # TMC to link aggregation  
        l_length = {}
        l_avgSpeed = {}
        tmc_edge_df = pd.DataFrame(link_tmc_dict.items(),columns = ['TMC','link'])
    
        
        for link in list(tmc_edge_df.link.unique()):
            l_xflows = pd.DataFrame()
            l_speed = pd.DataFrame()
            l_density = pd.DataFrame()
            
            tmc_list = tmc_edge_df[tmc_edge_df['link']==link]['TMC']
            df2 = df[df['tmc_code'].isin(tmc_list)]
            
            df2['Shape_Leng'] = 0.000621371192 * df2['Shape_Leng']  #since the speed (miles/hr) and len (meters --> miles)
            df2['prod'] = df2['xflow'] * df2['Shape_Leng'] / df2['avg_speed_day']
            df2['prod_speed'] = df2['Shape_Leng'] / df2['speed'] 
            df2['travelTime'] = df2['Shape_Leng'] / df2['avg_speed_day']
            
            grouped = df2.groupby('measurement_tstamp').sum()
            l_xflows['flow'] = grouped['prod'] / (grouped['travelTime'])
            l_speed['speed'] = grouped['prod_speed'] / (grouped['Shape_Leng'])
            
            if l_xflows.isnull().values.any() == True:
                l_xflows = l_xflows.interpolate(method='linear')
                
            if l_speed.isnull().values.any() == True:
                l_speed = l_speed.interpolate(method='linear')
                #break
                #time.sleep()
            summary = df2.groupby('measurement_tstamp').mean()
            tmc_length = df2.groupby('tmc_code').mean()['Shape_Leng']
            l_length[link] =  sum(tmc_length)
            tmc_avgSpeed = df2.groupby('tmc_code').mean()['speed']
            l_avgSpeed[link] = sum(tmc_avgSpeed*tmc_length)/sum(tmc_length)
            l_avgTravelTime = df2.groupby('measurement_tstamp').sum()['travel_time'] ### travel_time_minutes -> for 2015 , travel_time -> for 2012 
            
            l_avgTravelTime[l_avgTravelTime==0] = float('nan')
            
            if l_avgTravelTime.isnull().values.any() == True:
                l_avgTravelTime= l_avgTravelTime.interpolate(method='linear')
            
            
            link_flow[link] = l_xflows
            linkFlow = linkFlow.append(l_xflows)
            link_.extend([link]*len(l_xflows))
            
            speed_flow[link] = l_speed
            linkSpeed = linkSpeed.append(l_speed)
                        
            free_flow_tmc = free_flow_speed.reset_index()
            #free_flow_tmc = free_flow_tmc[free_flow_tmc['tmc_code'].isin(tmc_list)]
            free_flow_tmc = pd.merge(df2, free_flow_tmc, on='tmc_code', how='inner')
            free_flow_tmc = free_flow_tmc[~free_flow_tmc.tmc_code.duplicated(keep='first')]
            free_flow_tmc = free_flow_tmc.set_index('tmc_code')
            free_flow_tmc = sum(free_flow_tmc['free_flow_speed']*free_flow_tmc['Shape_Leng'])/sum(free_flow_tmc['Shape_Leng'])
            free_flow_link[link] = free_flow_tmc
            
            
            # calculate an average flow for that tmc and day 
            avg_xflows = pd.DataFrame()
            avg_flow = df2
            avg_flow = avg_flow.set_index('tmc_date')
            avg_flow = avg_flow[~avg_flow.index.duplicated(keep='first')]
            avg_flow = avg_flow.reset_index()
            #avg_flow = avg_flow.groupby('tmc_date').mean()
            avg_flow['prod'] = avg_flow['avg_flow'] * avg_flow['Shape_Leng'] / avg_flow['avg_speed_day']
            avg_flow['travelTime'] = avg_flow['Shape_Leng'] / avg_flow['avg_speed_day'] 
            grouped_avg_flow = avg_flow.groupby('measurement_tstamp').sum()
            avg_xflows['flow'] = grouped_avg_flow['prod'] / (grouped_avg_flow['travelTime'])
            link_avg_flow[str(link) + '_' + instance] = avg_xflows
            
            #calclate link capacity
            idx_r = result2.index.isin(tmc_list)
            link_att = result2[idx_r]
            str_inst = 'AB_' + instance + 'CAPAC'
            capacity_link[str(link) + '_' + instance] = sum(link_att[str_inst]*link_att['LENGTH'])/sum(link_att['LENGTH'])
            
            l_density = l_xflows
            l_density['density'] = greenshield_density(list(l_speed['speed']), capacity_link[str(link) + '_' + instance], free_flow_link[link], l_length[link], 1)
            l_density = pd.DataFrame(l_density['density'], columns=['density'])
            link_density[link] = l_density
            linkDensity = linkDensity.append(link_density[link])

            
        linkFlow['link'] = link_
        linkFlow = linkFlow.reset_index()
        unique_t = linkFlow['measurement_tstamp'].unique()
        
        linkSpeed['link'] = link_
        linkSpeed = linkSpeed.reset_index()
        #unique_t = linkFlow['measurement_tstamp'].unique()
        
        linkDensity['link'] = link_
        linkDensity = linkDensity.reset_index()
        
        G_[instance] = nx.DiGraph()
        for edge in list( G.edges()):
            G_[instance].add_edge(edge[0], edge[1], length = l_length[edge], avgSpeed = l_avgSpeed[edge])
        
        #CALCULATE FLOW AS DENSITY
        
        
        flow_after_conservation={}
        flow_before_conservation_ = {}
        speed_before_conservation = {}
        density_before_conservation = {}
        density_before_conservation_ = {}
        for idx in list(unique_t):
            ins = linkFlow[linkFlow['measurement_tstamp']==idx]
            ins = ins[['link','flow']].set_index('link').to_dict()['flow']
            ins_s = linkSpeed[linkSpeed['measurement_tstamp']==idx]
            ins_s = ins_s[['link','speed']].set_index('link').to_dict()['speed']
            ins_d = linkDensity[linkDensity['measurement_tstamp']==idx]
            ins_d['link'] = ins_d['link'].astype(str)
            ins_d = ins_d[['link','density']].set_index('link').to_dict()['density']
            
            
            if len(ins) == len(link_flow): #if there is no data of one llnk for an instance, then delete it
                flow_after_conservation[idx] = flow_conservation_adjustment(G,ins)
                flow_before_conservation_[idx] = ins
                
            if len(ins_s) == len(link_flow): #if there is no data of one llnk for an instance, then delete it
                speed_before_conservation[idx] = ins_s
                density_before_conservation[idx] = ins_d
                density_before_conservation_[idx] = flow_conservation_adjustment(G,ins_d )
                
        pd.to_pickle(flow_after_conservation, out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
        pd.to_pickle(linkFlow, out_dir + 'flows_before_QP' + files_ID + '_' + instance +'.pkz')
        pd.to_pickle(flow_before_conservation_, out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance +'.pkz')
        pd.to_pickle(speed_before_conservation, out_dir + 'speed_links' + files_ID + '_' + instance +'.pkz')
        pd.to_pickle(density_before_conservation_, out_dir + 'density_links' + files_ID + '_' + instance +'.pkz')
        pd.to_pickle(density_before_conservation, out_dir + 'density_links_before_QP' + files_ID + '_' + instance +'.pkz')
            
        for i in flow_after_conservation.keys():
            ts = pd.to_datetime(i) 
            d = ts.strftime('%Y-%m-%d-%H-%M-%S')
            flow_after_conservation[d] = flow_after_conservation.pop(i)
             
        for i in density_before_conservation.keys():
            ts = pd.to_datetime(i) 
            d = ts.strftime('%Y-%m-%d-%H-%M-%S')
            density_before_conservation[d] = density_before_conservation.pop(i)
        
        with open(out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.json', 'w') as fp:
            json.dump(flow_after_conservation, fp)

        with open(out_dir + 'density_links' + files_ID + '_' + instance +'.json', 'w') as afp:
            json.dump(density_before_conservation, afp)     
    
    zdump(capacity_link, out_dir + 'capacity_link' + files_ID + '.pkz')       
    zdump(link_avg_flow, out_dir + 'link_avg_day_flow' + files_ID + '.pkz')        
    zdump(G_, out_dir + 'G_' + files_ID + '.pkz' )
    zdump(free_flow_link, out_dir + 'free_flow_link' + files_ID + '.pkz' )
    return G_


