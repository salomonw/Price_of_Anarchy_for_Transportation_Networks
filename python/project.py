# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:13:13 2018

@author: Salomon Wollenstein
"""

' Packages '
import os
import sys
import numpy as np
import pandas as pd
import os 
import networkx as nx
import multiprocessing as mp
from dbfread import DBF    
import matplotlib.pyplot as plt
from gurobipy import *
import pickle

os.chdir('C:/Users/Salomon Wollenstein/Documents/GitHub/PoA_Jing_net/python')

get_ipython().magic(u'run utils.py')

'Parameters'
dir_shpfile = 'C:/Users/Salomon Wollenstein/Documents/GitHub/PoA/shp/Jing/journal.shp'
dir_data = 'E:/INRIX_byQuarter/4-6' # Will take all of the csv files contained in folders and subfolders
files_ID = '_MA_cdc_apr_'
dir_capacity_data = '../data/'

out_dir = '../tmp/'

# Filtering by date range and tmc
dates_input = [#{'id':'Jan','start_date': '2015-01-01' , 'end_date':'2015-01-10'}, 
               #{'id':'Feb','start_date': '2015-02-01' , 'end_date':'2015-02-15'}] 
               {'id':'Apr','start_date': '2015-04-01' , 'end_date':'2015-04-15'}] 
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

# ---------------------------------------------------------------------------------------------------------


#--------------------------- Preprocessing --------------------------------------

#Generating timestamp for resultsgi
import datetime 
now = datetime.datetime.now()
ts = now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

# Importing Jing data about the relation between TMC and link as well as the incidence matrix
os.chdir('C:/Users/Salomon Wollenstein/Documents/GitHub/InverseVIsTraffic/000_ETA/')
get_ipython().magic(u'run util_data_storage_and_load.py.py')
get_ipython().magic(u'run util.py')
from util import *  
    
os.chdir('C:/Users/Salomon Wollenstein/Documents/GitHub/InverseVIsTraffic/Python_files')
get_ipython().magic(u'run util_data_storage_and_load.py.py')
get_ipython().magic(u'run util.py')

from util import *      

link_with_capac_list = list(zload('C:/Users/Salomon Wollenstein/Documents/GitHub/InverseVIsTraffic/temp_files/links_with_capac.pkz'))

os.chdir('C:/Users/Salomon Wollenstein/Documents/GitHub/PoA_Jing_net/python')



shape = nx.read_shp(dir_shpfile)
edge_attributes = pd.DataFrame(i[2] for i in shape.edges(data=True))

tmc_edge_dict = dict(zip(list(shape.edges()), [shape.get_edge_data(list(shape.edges())[i][0],list(shape.edges())[i][1])['TMC']  for i in range(len(shape.edges()))]))

edge_tmc_dict = dict(zip([shape.get_edge_data(list(shape.edges())[i][0],list(shape.edges())[i][1])['TMC']  for i in range(len(shape.edges()))] , list(shape.edges())))


G = nx.DiGraph()
shapeG = nx.DiGraph()
tmc_net_list = []
tmc_att = pd.DataFrame()
idx = 0

link_tmc_dict = {}
for link in link_with_capac_list:
    init_node = link.init_node
    term_node = link.term_node
    G.add_edge(init_node,term_node)
    link_tmc = {tmc:(init_node,term_node) for tmc in link.tmc_set} 
    link_tmc_dict.update(link_tmc)
    for tmc in link.tmc_set:
        tmc_net_list.append(tmc)
        tmc_att = tmc_att.append(edge_attributes[edge_attributes['TMC']==tmc])

tmc_net_list = list(set(tmc_net_list))
    
# Visualize the topology of the network

edge_tmc_dict2 = dict((k, edge_tmc_dict[k]) for k in tmc_net_list)

shapeG = nx.DiGraph()        
shapeG.add_edges_from(edge_tmc_dict2.values())
        

node_coordinates = pd.DataFrame(i[0] for i in shapeG.nodes(data=True))
node_coordinates_name = dict(zip(shapeG.nodes(),[i for i in range(len(shapeG.nodes()))]))
shapeG = nx.relabel_nodes(shapeG, node_coordinates_name)
node_coordinates_dict = dict(zip(shapeG.nodes(),node_coordinates.values.tolist()))

nx.draw_networkx(shapeG, pos= node_coordinates_dict, node_size=30,font_size = 5)       
plt.savefig('network'+ files_ID +'.png', format='png', dpi=1200)

'''
dir_data = 'E:/INRIX_2015_trial_180531/New folder'

import multiprocessing as mp,os
import os 

def de(root,file, tmc_net_list,confidence_score_min,c_value_min):
    from os.path import basename
    import pandas as pd
    df = pd.DataFrame()
    cnt = 0
    iter_csv = pd.read_csv(root + '/' +  file, iterator=True, chunksize=200000)
    for chunk in iter_csv:
        chunk['measurement_tstamp']=pd.to_datetime(chunk['measurement_tstamp'], format='%Y-%m-%d %H:%M:%S')
        chunk = chunk.set_index('measurement_tstamp')
        df2 = filter_tmc(chunk,tmc_net_list,confidence_score_min,c_value_min)   
        df = df.append(df2)
        print('1fg')
    zdump(df,'../tmp/filtered_tmc_date_' + file[:-4]  +'.pkz')

#init objects
pool = mp.Pool(2)
jobs = []

#create jobs
dir_ = os.path.join(dir_data)
for root,dirs,files in os.walk(dir_):
    for file in files:
        if file.endswith(".csv"):
            jobs.append( pool.apply_async(de,(root ,file, tmc_net_list,confidence_score_min, c_value_min)))
           # jobs.append(mp.Process(target=de, args=(root ,file, tmc_net_list,confidence_score_min, c_value_min)))
#wait for all jobs to finish
for job in jobs:
    job.get()

#clean up
pool.close()
'''
#--------------------------- Preprocessing --------------------------------------

# Retrive the TMCs data

df = pd.DataFrame()
cnt = 0
filtered_files_list = []
dir_ = os.path.join(dir_data)
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
                filtered_files_list.append( '../tmp/filtered_tmc_date_' + file[:-4]  +'.pkz' )
                print(file + ' : ~' + str(percentage*100) + '%')
            print('-----------------------------------------------------')
            pd.to_pickle(df,'../tmp/filtered_tmc_' + file[:-4]  +'.pkz')
            del df


# Create list of files to read ( this can be used when you have passed the first filter of data)
filtered_files_list = []
dir_ = os.path.join(dir_data)
for root,dirs,files in os.walk(dir_):
    for file in files:
        if file.endswith(".csv"):
                filtered_files_list.append( '../tmp/filtered_tmc_' + file[:-4]  +'.pkz' )
                

# Read filtered data and create a file with all the relevant data 
df = pd.DataFrame()
for filtered_file in filtered_files_list:
    df1 = pd.read_pickle(filtered_file)
    df = df.append(df1)
    del df1
df.to_csv('../tmp/filtered_tmc' + files_ID + '.csv')

# Calculate percentiles 
tmc_free_flow = df.groupby('tmc_code').agg(percentile(percentile_free_flow))['speed'] 
tmc_free_flow.name= 'free_flow_speed'
pd.to_pickle(tmc_free_flow, '../tmp/free_flow_speed_ ' + files_ID + '.pkz')

# filtering the specific dates of interst and exporting csv file , can be avoided one you are working with the same dates
cnt = 0
df = pd.DataFrame()
for index, row in dates.iterrows():   
    iter_csv = pd.read_csv('../tmp/filtered_tmc' + files_ID + '.csv', iterator=True, chunksize=200000)
    for chunk in iter_csv:
        chunk['measurement_tstamp']=pd.to_datetime(chunk['measurement_tstamp'], format='%Y-%m-%d %H:%M:%S')
        chunk = chunk.set_index('measurement_tstamp')
        df2 = filter_dates(chunk, row['start_date'], row['end_date'])  
        df = df.append(df2)
        cnt = cnt + 1
        print(cnt)
    
    df.to_csv('../tmp/filtered_tmc_date' + files_ID +'.csv') 


# tmc and roadinv lookup'
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


df = pd.read_csv('../tmp/filtered_tmc_date' + files_ID + '.csv')


# Calculate reference speed table
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
    ref_speed_collection[tmc] = u
    



# filtering the time instances that the user is interested and exporting csv files for each of these
tmc_free_flow = pd.read_pickle('../tmp/free_flow_speed_ ' + files_ID + '.pkz')

for index, row in time_instances.iterrows():
    df = pd.DataFrame()
    cnt = 0
    iter_csv = pd.read_csv('../tmp/filtered_tmc_date' + files_ID + '.csv', iterator=True, chunksize=200000)
    for chunk in iter_csv:
        df2 = filter_time(chunk, row['start_time'], row['end_time'])  
        df = df.append(df2)
        cnt = cnt + 1
        print(cnt)        
    del df2
    
    # creating a table with the characteristics of TMCs in instance AM,MD,PM,NT    
    tmc_instance_stats = df.groupby('tmc_code').agg(np.mean)
    result2 = result.join(tmc_instance_stats, how='inner')
    result2 = pd.merge(lookup_tmc_roadinv, cap_data, right_index=True, left_index=True)
    result2 = result2.set_index('TMC')
    tmc_instance_char = tmc_instance_stats.join(result2, how='outer')
    tmc_instance_char = tmc_instance_char[~tmc_instance_char.index.duplicated(keep='first')]
    tmc_instance_char = tmc_instance_char.join(tmc_free_flow, how='outer')
    tmc_instance_char = tmc_instance_char.to_dict()
    pd.to_pickle(df, '../tmp/filtered_tmc_date_time' + files_ID + '_' + row['id'] +'.pkz')   
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
            row2['speed'] = speed   
        x_flow = greenshield(min(speed,free_flow_sp) , capacity , free_flow_sp)
        a.append([row2['idx'],x_flow])
    a = pd.DataFrame(a)
    a = a.rename(index=str, columns={0: "idx", 1 : "xflow"})
    df = df.join(a.set_index('idx'), on='idx')
    del a, chunk, 

    pd.to_pickle(df,'../tmp/filtered_tmc_date_time_flow' + files_ID + '_' + row['id'] +'.pkz')   
    
    del df
del cap_data, result, ref_speed_collection

zdump(ref_speed_collection,'../tmp/ref_speed_collection' + files_ID + '.pkz')
del ref_speed_collection    
    
# Filtering by weekdays/weekends
import time
link_flow = {}
linkFlow = pd.DataFrame()
link_ = list()
for instance in list(time_instances['id']):
    df = pd.read_pickle('../tmp/filtered_tmc_date_time_flow' + files_ID + '_' + instance +'.pkz')
    result2 = result2[~result2.index.duplicated(keep='first')]
    df = df.join(result2['LENGTH'], on = 'tmc_code', how = 'inner')
    if days_of_week == 'weekdays':
        df = df[df['dayWeek']>0]
        df = df[df['dayWeek']<7]
        
    # TMC to link aggregation
    
    tmc_edge_df = pd.DataFrame(link_tmc_dict.items(),columns = ['TMC','link'])
    for link in list(tmc_edge_df.link.unique()):
        l_xflows = pd.DataFrame()
        tmc_list = tmc_edge_df[tmc_edge_df['link']==link]['TMC']
        df2 = df[df['tmc_code'].isin(tmc_list)]
        df2['prod'] = df2['xflow'] * df2['LENGTH']
        grouped = df2.groupby('measurement_tstamp').sum()
        l_xflows ['flow'] = grouped['prod'] / grouped['LENGTH'] 
        if l_xflows.isnull().values.any() == True:
            time.sleep()
        l_length = sum(df2.groupby('tmc_code').mean()['LENGTH'])
        summary = df2.groupby('measurement_tstamp').mean()
        l_avgSpeed = summary['speed']
        l_avgTravelTime = df2.groupby('measurement_tstamp').sum()['travel_time_minutes']
        link_flow[link] = l_xflows 
        linkFlow = linkFlow.append(l_xflows)
        link_.extend([link]*len(l_xflows))
    
    linkFlow['link'] = link_
    linkFlow = linkFlow.reset_index()
    unique_t = linkFlow['measurement_tstamp'].unique()


    flow_after_conservation={}
    for idx in list(unique_t):
        ins = linkFlow[linkFlow['measurement_tstamp']==idx]
        ins = ins[['link','flow']].set_index('link').to_dict()['flow']
        flow_after_conservation[idx] = flow_conservation_adjustment(G,ins)
        a = flow_conservation_adjustment(G,ins)


    
   
    zdump(G,'../tmp/networkx_object_'+ files_ID + '.pkz')
    
    # use one of the edge properties to control line thickness
    edgewidth = np.divide(y,max(y))
    pos=nx.spectral_layout(G)
    nx.draw_networkx(G, pos, edge_labels= y , node_color='b' , edge_color= edgewidth, edge_cmap=plt.cm.Blues, node_size = 7 )
    plt.savefig('network_flows.png', format='png', dpi=1200)


#--------------------------- O-D Estuimation --------------------------------------


from itertools import combinations

# creating table of O-D pairs, in this case, all nodes are O-D pairs

OD_pairs = list(combinations(list(G.nodes())))






import json

link_label_dict = zload('../temp_files/link_label_dict_MA.pkz')
link_label_dict_ = zload('../temp_files/link_label_dict_MA_.pkz')

with open('../temp_files/link_length_dict_MA.json', 'r') as json_file:
    link_length_dict = json.load(json_file)

# number of links
m = 24

# number of routes (obtained by counting the rows with '->' in 'path-link_incidence.txt')
with open('../temp_files/path_link_incidence_MA_CDC.txt', 'r') as the_file:
    # path counts
    i = 0  
    for row in the_file:
        if '->' in row:
            i = i + 1
r = i

# number of O-D pairs
s = 8 * (8 - 1)

# initialize the path-link incidence matrix
A = np.zeros((m, r))

# read in the manually created path-link incidence file 
# create path-link incidence matrix A
with open('../temp_files/path_link_incidence_MA_CDC.txt', 'r') as the_file:
    # path counts
    i = 0  
    for row in the_file:
        if '->' in row:
            for j in range(m):
                if link_label_dict[str(j)] in row:
                    A[j, i] = 1
            i = i + 1
    assert(i == r)
zdump(A, '../temp_files/path-link_incidence_matrix_MA.pkz')

# link_length_dict['0'].length

# link_label_dict_

# read in the manually created path-link incidence file 
# calculate length of each route

length_of_route_list = []
with open('../temp_files/path_link_incidence_MA_CDC.txt', 'r') as the_file:
    for row in the_file:
        if '->' in row:
            link_list = []
            node_list = []
            for i in row.split('->'):
                node_list.append(int(i))
            for i in range(len(node_list))[:-1]:
                link_list.append('%d->%d' %(node_list[i], node_list[i+1]))
            length_of_route = sum([link_length_dict[str(link_label_dict_[link])] \
                                  for link in link_list])
            length_of_route_list.append(length_of_route)
zdump(length_of_route_list, '../temp_files/length_of_route_list_MA.pkz')

# length_of_route_list[139]

OD_pair_label_dict = zload('../temp_files/OD_pair_label_dict_MA.pkz')

# OD_pair_label_dict['(1, 2)']

# read in the manually created path-link incidence file 
# create label of each route
OD_pair_route_label_list = []
OD_pair_idx_list = []
route_idx_list = []
with open('../temp_files/path_link_incidence_MA_CDC.txt', 'r') as the_file:
    route_idx = 0
    for row in the_file:
        if '->' in row:
            node_list = []
            for i in row.split('->'):
                node_list.append(int(i))
            OD_pair_idx = OD_pair_label_dict[str((node_list[0], node_list[-1]))]
            OD_pair_idx_list.append(OD_pair_idx)
            route_idx_list.append(route_idx)
            OD_pair_route_label_list.append((OD_pair_idx, route_idx))
            route_idx += 1

OD_pair_route_dict = {}

for i in range(s):
    route_list = []
    for r_ in range(r):
        if OD_pair_idx_list[r_] == i:
            route_list.append(r_)
    OD_pair_route_dict[str(i)] = route_list
zdump(OD_pair_route_dict, '../temp_files/OD_pair_route_dict_MA.pkz')

# OD_pair_route_dict['6']


# calculate route choice probability matrix P
# logit choice parameter
theta = 0.8

P = np.zeros((s, r))
for i in range(s):
    for r in OD_pair_route_dict[str(i)]:
        P[i, r] = 1
        #P[i, r] = exp(- theta * length_of_route_list[r]) / \
        #            sum([exp(- theta * length_of_route_list[j]) \
        #                 for j in OD_pair_route_dict[str(i)]])
zdump(P, '../temp_files/OD_pair_route_incidence_MA.pkz')
# zdump(P, '../temp_files/logit_route_choice_probability_matrix_Sioux.pkz')

    
    
    
    
    
    
    
    
    
    
    

'TAKE AWAY THIS AND WIRTE IT UNDER THE UTILS SCRIPT'
def flow_conservation_adjustment(G,y):
    # y is a dict with flows and TMCs
    #from gurobipy import *

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
    for v in model.getVars():
        # print('%s %g' % (v.varName, v.x))
        u.append(v.x)
     #print('Obj: %g' % obj.getValue())
    return u

'''


















































# --------------------------------------------------------------------------------------


# Loading the TMC data by using the shapefile generated by QGIS
G = nx.read_shp(dir_shpfile)
isolate_nodes = list(nx.isolates(G))
G.remove_edges_from(isolate_nodes)
# Clean Isolate nodes and nodes with no outlinks ( recursevely)



node_coordinates = pd.DataFrame(i[0] for i in G.nodes(data=True))
node_coordinates_name = dict(zip(G.nodes(),[i for i in range(len(G.nodes()))]))
G = nx.relabel_nodes(G, node_coordinates_name)
node_coordinates_dict = dict(zip(G.nodes(),node_coordinates.values.tolist()))

edge_attributes = pd.DataFrame(i[2] for i in G.edges(data=True))

tmc_edge_dict = dict(zip(list(G.edges()), [G.get_edge_data(list(G.edges())[i][0],list(G.edges())[i][1])['TMC']  for i in range(len(G.edges()))]))




tmc_net_list = edge_attributes['TMC'].tolist()

# Visualize the topology of the network
nx.draw_networkx(G, pos= node_coordinates_dict, node_size=30,font_size = 5)       
plt.savefig('network'+ files_ID +'.png', format='png', dpi=1200)


df = pd.DataFrame()
cnt = 0

dir_ = os.path.join(dir_data)
for root,dirs,files in os.walk(dir_):
    for file in files:
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
                print(file + ' : ' + str(percentage) + '%')
                #print(file + ' : ' + str(cnt*200000 ) + ' rows analized')
            print('----------------------------')

df.to_csv('../tmp/filtered_tmc' + files_ID + '.csv') 

tmc_free_flow = df.groupby('tmc_code').agg(percentile(percentile_free_flow))['speed'] 
tmc_free_flow.name= 'free_flow_speed'
#tmc_free_flow_dict = tmc_free_flow.to_dict()

# filtering the specific dates of interst and exporting csv file 
df = pd.DataFrame()
cnt = 0
for index, row in dates.iterrows():   
    iter_csv = pd.read_csv('../tmp/filtered_tmc' + files_ID + '.csv', iterator=True, chunksize=200000)
    for chunk in iter_csv:
        chunk['measurement_tstamp']=pd.to_datetime(chunk['measurement_tstamp'], format='%Y-%m-%d %H:%M:%S')
        chunk = chunk.set_index('measurement_tstamp')
        df2 = filter_dates(chunk, row['start_date'], row['end_date'])  
        df = df.append(df2)
        cnt = cnt + 1
        print(cnt)
    
    df.to_csv('../tmp/filtered_tmc_date' + files_ID +'.csv') 


# tmc and roadinv lookup'
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

# filtering the time instances that the user is interested and exporting csv files for each of these
for index, row in time_instances.iterrows():
    df = pd.DataFrame()
    cnt = 0
    iter_csv = pd.read_csv('../tmp/filtered_tmc_date' + files_ID + '.csv', iterator=True, chunksize=200000)
    for chunk in iter_csv:
        df2 = filter_time(chunk, row['start_time'], row['end_time'])  
        df = df.append(df2)
        cnt = cnt + 1
        print(cnt)        
       
    #creating a table with the characteristics of TMCs in instance AM,MD,PM,NT    
    tmc_instance_stats = df.groupby('tmc_code').agg(np.mean)
    result2 = result.join(tmc_instance_stats, how='inner')
    result2 = pd.merge(lookup_tmc_roadinv, cap_data, right_index=True, left_index=True)
    result2 = result2.set_index('TMC')
    tmc_instance_char = tmc_instance_stats.join(result2, how='outer')
    tmc_instance_char = tmc_instance_char[~tmc_instance_char.index.duplicated(keep='first')]
    tmc_instance_char = tmc_instance_char.join(tmc_free_flow, how='outer')
    tmc_instance_char = tmc_instance_char.to_dict()

    #calculating flow data
    a=[]
    df['idx'] = range(1, len(df) + 1)
    for idx, row2 in df.iterrows():
        capacity = tmc_instance_char['AB_'+ time_instances.id[index] +'CAPAC'][row2['tmc_code']]
        free_flow_sp = tmc_instance_char['free_flow_speed'][row2['tmc_code']]
        speed = row2['speed']
        x_flow = greenshield(min(speed,free_flow_sp) , capacity , free_flow_sp)
        a.append([row2['idx'],x_flow])
        #df.set_value(row['idx'],'flow',x_flow)
    a = pd.DataFrame(a)
    a = a.rename(index=str, columns={0: "idx", 1 : "xflow"})
    df = df.join(a.set_index('idx'), on='idx')
        
    #tmc_instance_char.to_csv('../tmp/tmc_characteristics' + files_ID + '_' + row['id'] +'.csv') 
    
    df.to_csv('../tmp/filtered_tmc_date_time' + files_ID + '_' + row['id'] +'.csv')   
    
#del df
#del df2
 
df.reset_index(level=0, inplace=True)

link_edge = tmc_to_links(G)


link_edge_dict = dict(zip(link_edge.tmc, link_edge.link))
link_edge_df  = pd.DataFrame(data =[ link_edge_dict, dict((v,k) for k,v in tmc_edge_dict.iteritems())])
link_edge_df = link_edge_df.T
links_id = set(link_edge_df[0])

plt.figure()
for link in links_id:
    rdm_color = list(np.random.choice(range(256), size=1))
    nx.draw(G,pos=node_coordinates_dict, node_size=0.3,
                       edgelist= link_edge_df[link_edge_df[0]==link][1].tolist(),
                       width=.75,alpha=1,edge_color=random_color(),
                       font_size =5)
plt.savefig('destination_path_suff'+ files_ID +'.png', format='png', dpi=1200)




# Plot edges
plt.figure()
for link in links_id:
    rdm_color = list(np.random.choice(range(256), size=3))
    nx.draw_networkx_edges(G,pos=node_coordinates_dict, 
                       edgelist= link_edge_df[link_edge_df[0]==link][1].tolist(),
                       width=.75,alpha=1,edge_color=random_color())
plt.savefig('destination_path_suff'+ files_ID +'.png', format='png', dpi=1200)




# Playing with open street maps
import osmnx as ox
H = ox.graph_from_bbox(max(node_coordinates[1]),min(node_coordinates[1]),max(node_coordinates[0]),min(node_coordinates[0]), network_type='drive')
H_projected = ox.project_graph(H)
ox.plot_graph(H_projected)
plt.savefig('openstreetmap'+ files_ID +'.eps', format='eps', dpi=1200)
ec = ox.get_edge_colors_by_attr(H, attr='length')
ox.plot_graph(G, edge_color=ec)


df3 = (df.set_index('measurement_tstamp').groupby('tmc_code').resample(data_granularity).mean().reset_index())


df2 = df.pivot(index='measurement_tstamp', columns='tmc_code', values='xflow')
df2[index]=pd.to_datetime(df[i], format='%Y-%m-%d %H:%M:%S')    

y = df2.as_matrix()
y = np.transpose(y)
y= y[:,15]
### Generate for flow conservation
# G is a networkx graph
# y is a vector describing the flows and TMC id 

y_rand_flow = np.random.rand(len(G.edges()))*2400

y = dict(zip(edge_attributes.TMC,y_rand_flow))

def flow_conservation_adjustment(G,y,tmc_edge_dict):
    # y is a dict with flows and TMCs
    #from gurobipy import *
    tmc_edge_dict2 = dict((v,k) for k,v in tmc_edge_dict.iteritems())
    y_0 = y.values()
    model = Model("Flow_conservation_adjustment")
    l = len(y)
    x = []
    # Define variables (adjusted flows)
    for i in range(l):
        x.append(model.addVar(name = tmc_edge_dict.values()[i]))
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
            model.addConstr(quicksum(
                    model.getVarByName(G.get_edge_data(incoming_edge[0],incoming_edge[1])['TMC']) for incoming_edge in in_) == 
                    quicksum(model.getVarByName(G.get_edge_data(outgoing_edge[0],outgoing_edge[1])['TMC']) for outgoing_edge in out_))
            #model.addConstr(quicksum(y[G.get_edge_data(incoming_edge[0],incoming_edge[1])['TMC']]  for incoming_edge in in_) == quicksum(y[G.get_edge_data(outgoing_edge[0],outgoing_edge[1])['TMC']] for outgoing_edge in out_))
    
    
    model.update() 

    model.setParam('OutputFlag', False)
    model.optimize()

        
    u = []
    for v in model.getVars():
        # print('%s %g' % (v.varName, v.x))
        u.append(v.x)
    # print('Obj: %g' % obj.getValue())
    return u
            

tmc_to_links(G)


# From TMC to links
    
import networkx as nx
import multiprocessing as mp
from dbfread import DBF    
import matplotlib.pyplot as plt

get_ipython().magic(u'run utils.py')

'Parameters'
#dir_shpfile = 'C:/Users/Salomon Wollenstein/Documents/GitHub/PoA/shp/Toy_salo_2/Toy_salo_2.shp'
dir_shpfile = 'C:/Users/Salomon Wollenstein/Documents/GitHub/PoA/shp/Jing/journal.shp'
G = nx.read_shp(dir_shpfile)

  #  edges = G.edges(nbunch=random_node)
  #  tmc_ = nx.get_edge_attributes(G,'TMC')
  #  tmcs = []
   # for edge in edges:
   #     tmcs.append(tmc_[edge])
   # link = np.ones(len(edges))*link_id
    #df2 = pd.DataFrame(list(zip(link,tmcs)),columns=['link','tmc'])
    #df = df.append(df2)
    #del df2
   # set_ = diff(set_ , random_node)
               
            
          
            print 'node:'
            print 'incoming:'
            for incoming_edge in in_:
               y[G.get_edge_data(incoming_edge[0],incoming_edge[1])['TMC']]
                print G.get_edge_data(incoming_edge[0],incoming_edge[1])['TMC']
            print 'outgoing:'
            for outgoing_edge in out_:  
                 y[G.get_edge_data(outgoing_edge[0],outgoing_edge[1])['TMC']]
                print G.get_edge_data(outgoing_edge[0],outgoing_edge[1])['TMC']
          '''















