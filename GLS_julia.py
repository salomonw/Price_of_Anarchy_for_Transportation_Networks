'''
from utils_julia import *

def GLS_julia(xa, A, L):
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
    #Q_ = adj_PSD(Q_).real  # Ensure Q to be PSD
   
    Q = Q_

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



import numpy as np
from numpy.linalg import inv
import json
import collections 

G = zload(out_dir + 'G' + files_ID + '.pkz')
    
N = nx.incidence_matrix(G,oriented=True)
N = N.todense()

numEdges = len(G.edges())

#week_day_Apr_list = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]
#week_day_Apr_list = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27,30]

week_day_Apr_list = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 20]
#week_day_Apr_list = range(30)

for instance in list(time_instances['id']):
    #instance = 'AM'
    flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
    flow_after_conservation = pd.read_pickle(out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance +'.pkz')
    
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
    
    A = zload(out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')
    A = np.asmatrix(A)
    P = zload(out_dir + 'OD_pair_route_incidence_'+ instance + files_ID + '.pkz')
    P = np.asmatrix(P)
    
    L = np.size(P, 0) 
    
    x = np.nan_to_num(x)
    y = np.array(np.transpose(x))
    y = y[np.all(y != 0, axis=1)]
    x = np.transpose(y)
    x = np.matrix(x)
    
    L = np.size(P,0)  # dimension of xi
    
    #super_threshold_indices = P > 0.005
    #P5[super_threshold_indices] = 1
    
    xi_list = None
    try:
        xi_list = GLS(x, A, L)
    except:
        pass
    
    cnt_ = 0
    while xi_list == None:
        try:
            len_x = np.size(x,1)
            sample_size = np.random.randint(.5*len_x ,len_x)
            col_idx = np.random.choice(range(len_x), sample_size, replace=False)
            x1 = x[:,col_idx]
            cnt_ += 1
            print(cnt_)
            if cnt_ >= 45:
                xi_list = 1
                print('error, no PSD Q was found')
            xi_list = GLS(x1, A, L)

        except:
            pass
    
    
    
    lam_list = None
    try:
        lam_list = GLS2(x, A, P, L)
    except:
        pass
    
    cnt_ = 0
    while xi_list == None:
        try:
            len_x = np.size(x,1)
            sample_size = np.random.randint(.5*len_x ,len_x)
            col_idx = np.random.choice(range(len_x), sample_size, replace=False)
            x1 = x[:,col_idx]
            cnt_ += 1
            print(cnt_)
            if cnt_ >= 45:
                xi_list = 1
                print('error, no PSD Q was found')
            lam_list = GLS2(x1, A, P, L)

        except:
            pass 
    

    def saveDemandVec_(G, out_dir, instance, files_ID, lam_list, month_id ):
        lam_dict = {}
        n = len(G.nodes())  # number of nodes
        with open(out_dir + 'OD_demand_matrix_'+ month_id +'_weekday_'+ instance + files_ID + '.txt', 'w') as the_file:
            idx = 0
            for i in range(n + 1)[1:]:
                for j in range(n + 1)[1:]:
                    if i != j: 
                        key = str(idx)
                        lam_dict[key] = lam_list[idx]
                        the_file.write("%d,%d,%f\n" %(i, j, lam_list[idx]))
                        idx += 1

    #saveDemandVec_(G, out_dir, instance, files_ID, lam_list, month_id )
                        
'''

import collections
import numpy as np
from numpy.linalg import inv
import json
import pandas as pd
from utils_julia import *
from parameters import *


instance = 'AM'
# load logit_route_choice_probability_matrix
P = zload( out_dir + 'od_pair_route_incidence_' + instance + files_ID + '.pkz' )
P = np.matrix(P)

# print(np.size(P,0), np.size(P,1))

# load path-link incidence matrix
A = zload( out_dir + 'path-link_incidence_matrix_' + instance + files_ID + '.pkz' )

'''
# load link counts data
with open('../temp_files/link_day_minute_Apr_dict_JSON.json', 'r') as json_file:
    link_day_minute_Apr_dict_JSON = json.load(json_file)

week_day_Apr_list = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]

link_day_minute_Apr_list = []
for link_idx in range(24):
    for day in week_day_Apr_list: 
        for minute_idx in range(120):
            key = 'link_' + str(link_idx) + '_' + str(day)
            link_day_minute_Apr_list.append(link_day_minute_Apr_dict_JSON[key] ['PM_flow_minute'][minute_idx])

# print(len(link_day_minute_Apr_list))
'''

flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance + '.pkz')
flow_after_conservation = pd.read_pickle(out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance + '.pkz')

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
for link_idx in range(24):
    for day in range(31)[1:]: 
        key = 'link_' + str(link_idx) + '_' + str(day)
        link_day_Apr_list.append(link_day_minute_Apr_dict_JSON[key] ['PM_flow'])

# print(len(link_day_minute_Apr_list))

x_ = np.matrix(link_day_Apr_list)
x_ = np.matrix.reshape(x_, 24, 30)

x_ = np.nan_to_num(x_)
y_ = np.array(np.transpose(x_))
y_ = y_[np.all(y_ != 0, axis=1)]
x_ = np.transpose(y_)
x_ = np.matrix(x_)