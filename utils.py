# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:32:37 2018

@author: Salomon Wollenstein
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:39:40 2018

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
#from dbfread import DBF    
import matplotlib.pyplot as plt
from gurobipy import *
import pickle
#import pysal as ps
import pandas as pd
import numpy as np
import datetime

#Read dbf files as dataframes with pandas
def dbf2DF(dbfile, upper=True): #Reads in DBF files and returns Pandas DF
    db = ps.open(dbfile) #Pysal to open DBF
    d = {col: db.by_col(col) for col in db.header} #Convert dbf to dictionary
    #pandasDF = pd.DataFrame(db[:]) #Convert to Pandas DF
    pandasDF = pd.DataFrame(d) #Convert to Pandas DF
 #   if upper == True: #Make columns uppercase if wanted 
  #      pandasDF.columns = map(str.upper, db.header) 
    db.close() 
    return pandasDF

#Filtering by TMC
def filter_tmc(df,tmc_list,confidence_score_min,c_value_min): 
    df = df[df.tmc_code.isin(tmc_list)]
    df = df[df.confidence_score >= confidence_score_min]
    df2 = df[df.cvalue >= c_value_min]
    
    return df2

#Filtering between specific dates
def filter_dates(df,start_date,end_date):
    df = df[df.index >= start_date]
    df = df[df.index <= end_date]
    df_filter_data = df
    return df_filter_data

#Filerting between specific times 
def filter_time(df,start_time,end_time):
    df['measurement_tstamp']=pd.to_datetime(df['measurement_tstamp'], format='%Y-%m-%d %H:%M:%S')
    df= df.set_index('measurement_tstamp')
    df = df.between_time(start_time, end_time, include_start=True, include_end=True)
    df_filter_time = df
    return df_filter_time

#Redefining percentile function to use it on a pandas groupby
def percentile(n):
    def percentile_(x):
        return np.nanpercentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


#Green Shield model
def greenshield(speed,capacity,free_flow_speed):
    x = 4 * capacity * (np.true_divide(speed,free_flow_speed)-(np.true_divide(speed,free_flow_speed)**2))
    return x

# Plot shapefiles
def plot_shp(shp_obj):
    import shapefile as shp
    import matplotlib.pyplot as plt
    
    sf = shp_obj
    
    plt.figure()
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y)
    plt.show()

# Set differences between lists
def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def random_color():
    import random
    r = lambda: random.randint(0,255)
    return ('#%02X%02X%02X' % (r(),r(),r()))


def tmc_to_links(G):
    import networkx as nx
    from random import choice
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(columns=['link','tmc','roadnumb'])
    link_id = 0
    set_ = list(G.nodes())
    num_edges = len(G.edges())
    while len(df)<num_edges:  
        random_node = choice(set_)
        neighbors_set  = []
        #neighbors_set.append((random_node))
        if G.degree(random_node)<=2:
            neighbors_set.append(random_node)
            link_id += 1
            neighbors_set_iter = nx.all_neighbors(G,random_node)
            proposed_neighbors_set = list(neighbors_set_iter)
            increment_len_neighbors_set = 0 + len(neighbors_set)
            analized_neighbors = []
            analized_neighbors.append(random_node)
            new_proposed_2 =[]  
            while increment_len_neighbors_set > 0:
                #new_proposed_2 =[]   
                for neighbor in proposed_neighbors_set:
                    analized_neighbors.append(neighbor)
                   # neighbors_set.append((neighbor))
                                        
                    if G.degree(neighbor) <= 2:
                        neighbors_set.append((neighbor))
                        new_proposed = nx.all_neighbors(G,neighbor)
                        #new_proposed = diff(new_proposed,analized_neighbors)
                        
                        for nei in new_proposed:
                            new_proposed_2.append(nei)
                
                #proposed_neighbors_set.extend(new_proposed)
                proposed_neighbors_set.extend(new_proposed_2)
                proposed_neighbors_set = diff(proposed_neighbors_set, analized_neighbors)                        
                increment_len_neighbors_set = len(proposed_neighbors_set)
        
            edges = []
            
            #for i in neighbors_set:
            #     edges.append(G.edges(i)[0])
            edges.extend(list(G.in_edges(nbunch=neighbors_set)))
            edges.extend(list(G.out_edges(nbunch=neighbors_set)))
            
            edges = set(edges)
            
            #edges = G.edges(nbunch = neighbors_set)
            tmc_ = (nx.get_edge_attributes(G,'TMC'))
            road_ = 0
            tmcs = []
            road = []
            
            for edge in edges:
                tmcs.append(tmc_[edge])
                road.append(road_)
            
            link = np.ones(len(edges)) * link_id
            df2 = pd.DataFrame(list(zip(link,tmcs,road)),columns=['link','tmc','roadnumb'])
            df = df.append(df2)
            #df = df.drop_duplicates('tmc')
            del df2
        
        df = df.reset_index(drop=True)
        # Dealing with tmcs that connect two intestections
        if min(dict(nx.degree(G,nbunch=set_)).values()) > 2:
            edges = []
            edges.extend(list(G.in_edges(nbunch=set_)))
            edges.extend(list(G.out_edges(nbunch=set_)))
            
            for edge in edges:
                tmcs = []
                road = []
                tmcs.append(tmc_[edge])
                road.append(road_[edge])
                link += 1
                df2 = pd.DataFrame(list(zip(link,tmcs,road)),columns=['link','tmc','roadnumb'])
                df = df.append(df2)
                #df = df.drop_duplicates('tmc')
                del df2
            df = df.reset_index(drop=True)                
            df = df.groupby('tmc', group_keys=False).apply(lambda x: x.loc[x.link.idxmin()])
        set_ = diff(set_ , neighbors_set)
        #set_ = diff(set_ , [random_node])
    return df



# define a function converting rough flow vector to feasible flow vector 
# (satisfying flow conservation law)
def flow_conservation_adjustment(y_0):
    L = len(y_0)  # dimension of flow vector x
    assert(L == 24)

    # y_0 = x[:,1]  # initial flow vector

    model = Model("Flow_conservation_adjustment")

    y = []
    for l in range(L):
        y.append(model.addVar(name='y_' + str(l)))

    model.update() 

    # Set objective: ||y-y_0||^2
    obj = 0
    for l in range(L):
        obj += (y[l] - y_0[l]) * (y[l] - y_0[l])
    model.setObjective(obj)

    # Add nonnegative constraint: y >= 0
    for l in range(L):
        model.addConstr(y[l] >= 0)
    # Add flow conservation constraints
    model.addConstr(y[1] + y[3] == y[0] + y[2])
    model.addConstr(y[0] + y[5] + y[7] == y[1] + y[4] + y[6])
    model.addConstr(y[2] + y[4] + y[9] + y[11] == y[3] + y[5] + y[8] + y[10])
    model.addConstr(y[6] + y[13] + y[17] == y[7] + y[12] + y[16])
    model.addConstr(y[8] + y[12] + y[15] + y[19] == y[9] + y[13] + y[14] + y[18])
    model.addConstr(y[10] + y[14] + y[21] == y[11] + y[15] + y[20])
    model.addConstr(y[18] + y[20] + y[23] == y[19] + y[21] + y[22])
    model.addConstr(y[16] + y[22] == y[17] + y[23])

    model.update() 

    model.setParam('OutputFlag', False)
    model.optimize()

    y = []
    for v in model.getVars():
        # print('%s %g' % (v.varName, v.x))
        y.append(v.x)
    # print('Obj: %g' % obj.getValue())
    return y



# Data Storage and Load
# These two functions "zdump" and "zload" were written by Jing Wang
# cf. https://github.com/hbhzwj/GAD/blob/master/gad/util/util.py

try:
    import cPickle as pickle
except ImportError:
    import pickle
import gzip
proto = pickle.HIGHEST_PROTOCOL

def zdump(obj, f_name):
    f = gzip.open(f_name,'wb', proto)
    pickle.dump(obj,f)
    f.close()

def zload(f_name):
    f = gzip.open(f_name,'rb', proto)
    obj = pickle.load(f)
    f.close()
    return obj

def dump(obj, f_name):
    f = open(f_name,'wb', proto)
    pickle.dump(obj,f)
    f.close()

def load(f_name):
    f = open(f_name,'rb', proto)
    obj = pickle.load(f)
    f.close()
    return obj




