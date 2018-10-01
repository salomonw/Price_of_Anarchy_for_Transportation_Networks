
import collections
import numpy as np
from numpy.linalg import inv
import json
import pandas as pd
from utils_julia import *
from parameters_julia import *

def GLS_juliaf(instance):

    od_nodesList_dict = zload(out_dir + 'od_nodesList_dict.pkz')


    # load logit_route_choice_probability_matrix
    P = zload( out_dir + 'od_pair_route_incidence_' + instance + files_ID + '.pkz' )
    P = np.matrix(P)

    # load path-link incidence matrix
    A = zload( out_dir + 'path-link_incidence_matrix_' + instance + files_ID + '.pkz' )


    link_edge_dict = zload( out_dir + 'link_edge_dict' + files_ID + '.pkz' )


    with open(out_dir + 'link_label_dict_' + '.json', 'r') as json_file:
            link_label_dict_ = json.load(json_file)

    numEdges = len(link_label_dict_)

    week_day_Apr_list = week_day_list

    with open(out_dir + 'link_min_dict'+ files_ID + '.json', 'r') as json_file:
            link_day_minute_Apr_dict_JSON = json.load(json_file)

    numDays = len(week_day_Apr_list)

    flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance + '.pkz')
    #flow_after_conservation = pd.read_pickle(out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance + '.pkz')

    flow_after_conservation = collections.OrderedDict(sorted(flow_after_conservation.items()))

    x = np.zeros(numEdges)
    for ts in flow_after_conservation :
        #ts = flow_after_conservation.keys()[0]
        #x = np.zeros(numEdges)
        day = (ts.astype('datetime64[D]') - ts.astype('datetime64[M]') + 1).astype(int)
        if np.isin(day, week_day_Apr_list)+0 == np.int32(1) :
            a = np.array(list(flow_after_conservation[ts].values()))
            x = np.c_[x,a]

    

    x = np.delete(x,0,1)
    x = np.asmatrix(x)


    x = np.nan_to_num(x)
    y = np.array(np.transpose(x))
    y = y[np.all(y != 0, axis=1)]
    x = np.transpose(y)
    x = np.matrix(x)

    link_day_Apr_list = []
    #year = 2012
    #month = 4
    link_vector = []
    for edge in link_edge_dict.values():
        #print edge
        for day in week_day_Apr_list: 
            key = 'link_' +  str(edge) + '_' + str(year) + '_' + str(month) + '_'   + str(day) 
        #    print key
            link_day_Apr_list.append(link_day_minute_Apr_dict_JSON[key] ['avg_flow_' + instance ])
        link_vector.append(edge)

    # print(len(link_day_minute_Apr_list))

    x_ = np.matrix(link_day_Apr_list)
    x_ = np.matrix.reshape(x_, numEdges, numDays)

    x_ = np.nan_to_num(x_)
    y_ = np.array(np.transpose(x_))
    y_ = y_[np.all(y_ != 0, axis=1)]
    x_ = np.transpose(y_)
    x_ = np.matrix(x_)
    #it = flow_after_conservation.items()
    return x_, link_vector