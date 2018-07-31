# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 23:39:57 2018

@author: Salomon Wollenstein
"""
from utils import *

def parse_data_for_Julia(out_dir, files_ID, time_instances):
    import collections
    import json
    G_ = zload(out_dir + 'G_' + files_ID + '.pkz' )
    free_flow_link = zload(out_dir + 'free_flow_link' + files_ID + '.pkz' )
    G = G_[time_instances['id'][0]]
    link_avg_flow = zload(out_dir + 'link_avg_day_flow' + files_ID + '.pkz')
    capacity_link = zload(out_dir + 'capacity_link' + files_ID + '.pkz')
    
    edges = list(G.edges())
    
    link_edge_dict = {}
    flows_after={}
    flows_before={}
    link_min_dict = {}
    idx = 0
    for edge in edges:
        
        from_ = edge[0]
        to_ = edge[1]
        
        key = {}
        edge_len = G.get_edge_data(from_,to_)['length']
        free_flow_speed = free_flow_link[edge]
        key['avg_speed'] = G.get_edge_data(from_,to_)['avgSpeed']
        key['edge_len'] = edge_len
        key['free_flow_speed'] = free_flow_speed
        key['free_flow_time'] = edge_len/free_flow_speed
        key['init_node'] = from_
        key['term_node'] = to_
        
        for instance in time_instances['id']:   
            flows_before = pd.read_pickle(out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance +'.pkz')
            flows_before = collections.OrderedDict(sorted(flows_before.items()))
            flows_after =  pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
            flows_after = collections.OrderedDict(sorted(flows_after.items()))
            
            flows_after_k = list(flows_after.keys())
            flows_after_k = pd.DatetimeIndex(flows_after_k)
            
            uniq_days = flows_after_k.normalize().unique()
            
            for i in uniq_days:
                day = np.asscalar(pd.DatetimeIndex([i]).day)
                month = np.asscalar(pd.DatetimeIndex([i]).month)
                year = np.asscalar(pd.DatetimeIndex([i]).year)            
                keys = 'link_' + str(edge) + '_' + str(year) + '_' + str(month) + '_' + str(day)
                from_date=datetime.datetime(year,month,day,0,0,0)
                to_date=from_date+datetime.timedelta(days=1)
                
                idx2=(flows_after_k>from_date) & (flows_after_k<=to_date)
                flow_days = flows_after_k[idx2]
            
                flow_days_dict = { your_key: flows_after[your_key] for your_key in flow_days.values }
                
                flow_avg = link_avg_flow[str(edge) + '_' + instance]
                idx3 = (flow_avg.index>from_date) & (flow_avg.index<=to_date)
                flow_avg = flow_avg[idx3]
                key['avg_flow_' + instance] = flow_avg['flow'][0]
                
                key['capac_'+ instance] = capacity_link[str(edge) + '_' + instance]
                
                v = []
                for j in flow_days_dict:
                    
                    link_min_dict[keys] = key
                    v.append(flow_days_dict[j][str(edge)])
    
            key['flow_'+instance] = v
            link_min_dict[keys] = key
                    #link_min_dict[keys] = key
        idx += 1
        link_edge_dict[idx] = edge
        print('link: ' + str(edge) + ' has been processed ' + str(idx)  + ' !')
    
    with open(out_dir + 'link_edge_dict' + files_ID  + '.json', 'w') as fp:
            json.dump(link_edge_dict, fp)
    
    with open(out_dir + 'link_min_dict' + files_ID  + '.json', 'w') as fp:
            json.dump(link_min_dict, fp)
    
    zdump(link_edge_dict, out_dir + 'link_edge_dict' + files_ID + 'pkz')
    zdump(link_min_dict, out_dir + 'link_min_dict' + files_ID + 'pkz')
    



#### ------
    