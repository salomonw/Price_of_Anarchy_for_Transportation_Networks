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

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

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
    S = adj_PSD(S)
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
            df2['Shape_Leng'] = 0.000621371192 * df2['Shape_Leng']  #since the speed (miles/hr) and len (meters --> miles)
            df2['prod'] = df2['xflow'] * df2['Shape_Leng'] / df2['speed']
            df2['travelTime'] = df2['Shape_Leng'] / df2['speed']
            grouped = df2.groupby('measurement_tstamp').sum()
            l_xflows['flow'] = grouped['prod'] / (grouped['travelTime'])
            
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

        G_[instance] = nx.DiGraph()
        for edge in list( G.edges()):
            G_[instance].add_edge(edge[0], edge[1], length = l_length[edge], avgSpeed = l_avgSpeed[edge])
            
        flow_after_conservation={}
        for idx in list(unique_t):
            ins = linkFlow[linkFlow['measurement_tstamp']==idx]
            ins = ins[['link','flow']].set_index('link').to_dict()['flow']
            if len(ins) == len(link_flow): #if there is no data of one llnk for an instance, then delete it
                flow_after_conservation[idx] = flow_conservation_adjustment(G,ins)
            
        pd.to_pickle(flow_after_conservation, out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
    zdump(G_, out_dir + 'G_' + files_ID + '.pkz' )
    return G_
    

#### ------------ OD Estimation ------------- :

def od_pair_definition(out_dir, files_ID ):
    G = zload(out_dir + 'G' + files_ID + '.pkz')
    od_pairs=[]
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                od_pairs.append((i, j))
    zdump(od_pairs, out_dir + 'od_pairs'+ files_ID + '.pkz')


def routes(G, out_dir, files_ID, od_pairs, number_of_routes_per_od, instance):
    
    # Create Routes
    routes = []
    link_dict = dict(zip(list(G.edges()),range(len(list(G.edges())))))
    OD_dict = dict(zip(od_pairs,range(len(od_pairs))))
    OD_pair_route_dict = {}
    cnt_od = 0
    cnt_route = 0
    for od in od_pairs:
        route = nx.all_simple_paths(G,od[0],od[1])
        route = list(route)
        route_length_od = {}
        for r in route:
            total_length = 0
            total_travelTime = 0
            for i in range(len(r)-1):
                source, target = r[i], r[i+1]
                edge = G[source][target]
                length = edge['length']
                total_length += length
                travelTime = edge['length']/edge['avgSpeed']
                total_travelTime += travelTime 
            #routes.append(r)
            route_length_od[tuple(r)] = [total_length, total_travelTime]
            filtered_routes = sorted(route_length_od.iteritems(), key=lambda (k,v): (v,k))[:number_of_routes_per_od]
        
        OD_pair_route_list = []
        for i in filtered_routes:
            routes.append([i[0], i[1], cnt_route])
            OD_pair_route_list.append(cnt_route)
            cnt_route += 1
        OD_pair_route_dict[cnt_od] = OD_pair_route_list
        cnt_od += 1
            
        # Create Path-Link Incidence Matrix
        r = len(routes) # Number of routes
        m = len(list(G.edges())) # Number of links
        A = np.zeros((m, r))
        for route2 in routes:
            edgesinpath=zip(route2[0][0:],route2[0][1:])           
            for edge in edgesinpath:
                #print(link)
                link = link_dict[edge] 
                A[link,routes.index(route2)] = 1
                
    zdump(A, out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')

    #length_of_route_list = [[i[1][0], i[2]] for i in routes]
    length_of_route_dict = {}
    for i in routes:
        length_of_route_dict[i[2]]=i[1][0] 
    # calculate route choice probability matrix P
    # logit choice parameter
    theta = 0.8 #send as parameter !
    s = len(od_pairs) # number of OD pairs
    r = len(routes) # Number of routes
    P = np.zeros((s, r))
    for i in range(s):
        for r_ in OD_pair_route_dict[i]:
            P[i, r_] = 1
            
            P[i, r_] = exp(- theta * length_of_route_dict[r_]) / \
            sum([exp(- theta * length_of_route_dict[j]) \
                             for j in OD_pair_route_dict[i]])
    zdump(P, out_dir + 'OD_pair_route_incidence_'+ instance + files_ID + '.pkz')     
      
    #return routes, A, P

#number_of_routes_per_od = 3
#routes = routes(G, od_pairs, number_of_routes_per_od)



def path_incidence_matrix(out_dir, files_ID, time_instances, number_of_routes_per_od, theta ):
    G_ = zload( out_dir + 'G_' + files_ID + '.pkz' )
    od_pairs = zload(out_dir + 'od_pairs'+ files_ID + '.pkz')
    
    for instance in list(time_instances['id']):
        G = G_[instance]
        routes(G, out_dir, files_ID, od_pairs, number_of_routes_per_od, instance)


#def path_link_incidence_matrix
# Create Path-Link Incidence Matrix

'''
G = zload(out_dir + 'G' + files_ID + '.pkz')

N = nx.incidence_matrix(G,oriented=True)

# load link_route incidence matrix
N = nx.incidence_matrix(G,oriented=True)
N = N.todense()


numEdges = len(G.edges())
instance = 'AM'
flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
x = np.zeros(numEdges)

for ts in flow_after_conservation :
    #x = np.zeros(numEdges)
    a = np.array(list(flow_after_conservation[ts].values()))
    x = np.c_[x,a]

x = np.delete(x,0,1)
x = np.asmatrix(x)
A = zload(out_dir + 'path-link_incidence_matrix'+ instance + files_ID + '.pkz')
A = np.asmatrix(A)
P = zload(out_dir + 'OD_pair_route_incidence'+ instance + files_ID + '.pkz')
P = np.asmatrix(P)

L = np.size(P, 1) 
'''

# implement GLS method to estimate OD demand matrix


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


def GLS(xa, A, L):
    import numpy as np
    from numpy.linalg import inv
    import json
    """
    x: sample matrix, each column is a link flow vector sample; 24 * K
    A: path-link incidence matrix
    P: logit route choice probability matrix
    L: dimension of xi
    ----------------
    return: xi
    ----------------
    """
    K = np.size(xa, 1)
    S = samp_cov(xa)

    inv_S = inv(S).real

    A_t = np.transpose(A)

    Q_ = np.dot(np.dot(A_t, inv_S), A)
    Q = adj_PSD(Q_).real  # Ensure Q to be PSD
    #Q = Q_

    b = sum([np.dot(np.dot(A_t, inv_S), xa[:, k]) for k in range(K)])

    model = Model("OD_matrix_estimation")

    xi = []
    for l in range(L):
        xi.append(model.addVar(name='xi_' + str(l)))

    model.update() 

    # Set objective: (K/2) xi' * Q * xi - b' * xi
    obj = 0
    for i in range(L):
        for j in range(L):
            obj += (1.0 / 2) * K * xi[i] * Q[i, j] * xi[j]
    for l in range(L):
        obj += - b[l] * xi[l]
    model.setObjective(obj)

    # Add constraint: xi >= 0
    for l in range(L):
        model.addConstr(xi[l] >= 0)
   
    model.update() 

    model.setParam('OutputFlag', False)
    model.optimize()

    xi_list = []
    for v in model.getVars():
        xi_list.append(v.x)
 
    return xi_list

def ODandRouteChoiceMat(P, xi_list):
    model = Model("OD_matrix_and_route_choice_matrix")
    
    L = np.size(P,0)  # dimension of lam
    
    lam = []
    for l in range(L):
        lam.append(model.addVar(name='lam_' + str(l)))
        model.update()
        model.addConstr(lam[l] >= 0)
        
    p = {}
    for i in range(np.size(P,0)):
        for j in range(np.size(P,1)):
            p[(i,j)] = model.addVar(name='p_' + str(i) + ',' + str(j))  
            model.update()
            model.addConstr(p[(i,j)] >= 0)
            if P[i,j] == 0:
                model.addConstr(p[(i,j)] == 0)
    
    for i in range(np.size(P,0)):
        model.addConstr(sum([p[(i,j)] for j in range(np.size(P,1))]) == 1)
    
    for idx in range(len(xi_list)):
        model.addConstr(sum([p[(l,idx)] * lam[l] for l in range(L)]) >= xi_list[idx])
        model.addConstr(sum([p[(l,idx)] * lam[l] for l in range(L)]) <= xi_list[idx])
    
    model.update()
    
    obj = 1
    model.setObjective(obj)
    
    model.update() 
    
    model.setParam('OutputFlag', False)
    model.optimize()
    
    lam_list = []
    for v in model.getVars():
        # print('%s %g' % (v.varName, v.x))
        if 'lam' in v.varName:
            lam_list.append(v.x)
            
    return lam_list

                    
def runGLS(out_dir, files_ID, time_instances):
    import numpy as np
    from numpy.linalg import inv
    import json
    
    G = zload(out_dir + 'G' + files_ID + '.pkz')
        
    N = nx.incidence_matrix(G,oriented=True)
    N = N.todense()
    
    numEdges = len(G.edges())
    
    for instance in list(time_instances['id']):
        #instance = 'AM'
        flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
        x = np.zeros(numEdges)
        
        for ts in flow_after_conservation :
            #x = np.zeros(numEdges)
            a = np.array(list(flow_after_conservation[ts].values()))
            x = np.c_[x,a]
        
        x = np.delete(x,0,1)
        x = np.asmatrix(x)
        A = zload(out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')
        A = np.asmatrix(A)
        P = zload(out_dir + 'OD_pair_route_incidence_'+ instance + files_ID + '.pkz')
        P = np.asmatrix(P)
        
        L = np.size(P, 1) 
        
        x = np.nan_to_num(x)
        y = np.array(np.transpose(x))
        y = y[np.all(y != 0, axis=1)]
        x = np.transpose(y)
        x = np.matrix(x)
          
        L = np.size(P,1)  # dimension of xi
        
        xi_list = GLS(x, A, L)
        
        def saveDemandVec(G, out_dir, instance, files_ID, lam_list):
            lam_dict = {}
            n = len(G.nodes())  # number of nodes
            with open(out_dir + 'OD_demand_matrix_Apr_weekday_'+ instance + files_ID + '.txt', 'w') as the_file:
                idx = 0
                for i in range(n + 1)[1:]:
                    for j in range(n + 1)[1:]:
                        if i != j: 
                            key = str(idx)
                            lam_dict[key] = lam_list[idx]
                            the_file.write("%d,%d,%f\n" %(i, j, lam_list[idx]))
                            idx += 1
        saveDemandVec(G, out_dir, instance, files_ID, xi_list)
       
    
    
    
    
    
 
    '''
    xi_dict = {}

    L, len(xi_list), xi_list
    
    range(31)[1:]
    
    link_day_Apr_list = []
    for link_idx in range(24):
        for day in range(31)[1:]: 
            key = 'link_' + str(link_idx) + '_' + str(day)
            link_day_minute_Apr_list.append(link_day_minute_Apr_dict_JSON[key] ['PM_flow'])
    
    # print(len(link_day_minute_Apr_list))
    
    x_ = np.matrix(link_day_Apr_list)
    x_ = np.matrix.reshape(x_, 24, 30)
    
    x_ = np.nan_to_num(x_)
    y_ = np.array(np.transpose(x_))
    y_ = y_[np.all(y_ != 0, axis=1)]
    x_ = np.transpose(y_)
    x_ = np.matrix(x_)

    
   
    
    
    
    
    # load link counts data
    with open('../temp_files/link_day_minute_Apr_dict_JSON.json', 'r') as json_file:
        link_day_minute_Apr_dict_JSON = json.load(json_file)
    
    # week_day_Apr_list = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]
    week_day_Apr_list = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23]
    
    link_day_minute_Apr_list = []
    for link_idx in range(24):
        for day in week_day_Apr_list: 
            for minute_idx in range(120):
                key = 'link_' + str(link_idx) + '_' + str(day)
                link_day_minute_Apr_list.append(link_day_minute_Apr_dict_JSON[key] ['NT_flow_minute'][minute_idx])
    
    # print(len(link_day_minute_Apr_list))
    
    x = np.matrix(link_day_minute_Apr_list)
    x = np.matrix.reshape(x, 24, 120 * len(week_day_Apr_list))
    
    x = np.nan_to_num(x)
    y = np.array(np.transpose(x))
    y = y[np.all(y != 0, axis=1)]
    x = np.transpose(y)
    x = np.matrix(x)
    
    # print(np.size(x,0), np.size(x,1))
    # print(x[:,:2])
    # print(np.size(A,0), np.size(A,1))
    
    L = np.size(P,1)  # dimension of xi
    
    xi_list = GLS(x, A, L)
    
    # write estimation result to file
    def saveDemandVec(lam_list):
        lam_dict = {}
        n = 8  # number of nodes
        with open('../temp_files/OD_demand_matrix_Apr_weekday_NT.txt', 'w') as the_file:
            idx = 0
            for i in range(n + 1)[1:]:
                for j in range(n + 1)[1:]:
                    if i != j: 
                        key = str(idx)
                        lam_dict[key] = lam_list[idx]
                        the_file.write("%d,%d,%f\n" %(i, j, lam_list[idx]))
                        idx += 1
    














def routeLength(route):
    link_list = []
    node_list = []
    for i in route.split('->'):
        node_list.append(int(i))
    for i in range(len(node_list))[:-1]:
        link_list.append('%d->%d' %(node_list[i], node_list[i+1]))
    length_of_route = sum([link_length_dict[str(link_label_dict_[link])] for link in link_list])
    return length_of_route


def find_paths_of_OD_pairs(G, od_pairs):
    





#def createPathLinkIncidenceMatrix(G, link_length, link_freeFlowTravelTime):
#    nodes = list(G.nodes())
#    neighbors={}
#    for node in nodes:
#        neighbors_dict = (G.neighbors(node))
      
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
'''
