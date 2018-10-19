
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:00:57 2018

@author: Salomon Wollenstein
"""
import os
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt

os.chdir('G:/My Drive/Github/InverseVIsTraffic/Python_files/') # CHANGES 
execfile('util_data_storage_and_load.py')
execfile('util.py')

from util import *   
from util_data_storage_and_load import * 

def import_jing_net(dir_shpfile, files_ID, out_dir):
    # Importing Jing data about the relation between TMC and link as well as the incidence matrix
    os.chdir('G:/My Drive/Github/InverseVIsTraffic/Python_files')
    link_with_capac_list = list(zload('G:/My Drive/Github/InverseVIsTraffic1/temp_files/links_with_capac.pkz'))
    
    os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')
    
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
    

    zdump(tmc_net_list, out_dir + 'tmc_net_list' + files_ID + '.pkz')
    zdump(link_tmc_dict, out_dir + 'link_tmc_dict' + files_ID + '.pkz')
    zdump(tmc_att, out_dir + 'tmc_att' + files_ID + '.pkz')
    # Visualize the topology of the network
    
    edge_tmc_dict2 = dict((k, edge_tmc_dict[k]) for k in tmc_net_list)
    
    shapeG = nx.DiGraph()        
    shapeG.add_edges_from(edge_tmc_dict2.values())
            
    
    node_coordinates = pd.DataFrame(i[0] for i in shapeG.nodes(data=True))
    node_coordinates_name = dict(zip(shapeG.nodes(),[i for i in range(len(shapeG.nodes()))]))
    shapeG = nx.relabel_nodes(shapeG, node_coordinates_name)
    node_coordinates_dict = dict(zip(shapeG.nodes(),node_coordinates.values.tolist()))
    
    nx.draw_networkx(shapeG, pos= node_coordinates_dict, node_size=30,font_size = 5)       
    plt.savefig(out_dir + 'network_tmc_topology'+ files_ID +'.png', format='png', dpi=1200)
    
    zdump(G, out_dir + 'G' + files_ID + '.pkz')
    
    return G

os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')