from parameters_julia import *
from utils_julia import *
from OD_functions import *
#from functions import *
import numpy as np
from numpy.linalg import inv
import json
import collections 
import math

def average_over_time(time_window, x):
   # x = np.transpose(x)
    len_x_ = len(x)
    len_x = len(np.transpose(x))
    it = int(math.floor(len_x/time_window))
    y = zeros(len_x_)
    for i in range(it):
        x_ = np.mean(x[:, range(i*time_window,(i+1)*time_window)],axis=1)
        x_ = np.matrix(x_)
        y = np.c_[y,x_]
    y = np.delete(y,0,1)
    y = np.asmatrix(y)
    return y
    
def GLS(x, A, L):
    """
    x: sample matrix, each column is a link flow vector sample; 24 * K
    A: path-link incidence matrix
    P: logit route choice probability matrix
    L: dimension of xi
    ----------------
    return: xi
    ----------------
    """
    K = np.size(x, 1)
    S = samp_cov(x)
    
    inv_S = inv(S).real
    
    A_t = np.transpose(A)

    Q_ = np.dot(np.dot(A_t, inv_S), A)

    Q = Q_

    T = [np.dot(np.dot(A_t, inv_S), x[:, k]) for k in range(K)]
    b = [sum(i) for i in zip(*T)]


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

        #model.addConstr(xi[l] == 0)
    model.update() 

    model.setParam('OutputFlag', False)
    model.optimize()

    xi_list = []
    for v in model.getVars():

        xi_list.append(v.x)

    return xi_list

def saveDemandVec(edges, out_dir, instance, files_ID, lam_list, month_id, day ):
    lam_dict = {}
    n = edges  # number of nodes
    with open(out_dir + 'OD_demands/OD_demand_matrix_'+ month_id + '_' + day +  '_weekday_'+ instance + files_ID + '.txt', 'w') as the_file:
        idx = 0
        for i in range(n + 1)[1:]:
            for j in range(n + 1)[1:]:
                if i != j: 
                    key = str(idx)
                    lam_dict[key] = lam_list[idx]
                    the_file.write("%d,%d,%f\n" %(i, j, lam_list[idx]))
                    idx += 1

def runGLS_f(out_dir, files_ID, time_instances, month_w, week_day_list, average_over_time):

    time_window = average_over_time
    edges = zload(out_dir + 'link_length_dict.pkz')
    node_link_inc = zload(out_dir + 'node_link_incidence.pkz')    
    numEdges = len(edges)
    numNodes = len(node_link_inc)
    

    week_day_Apr_list = week_day_list
    
    for instance in time_instances['id']:
        x_ins = np.zeros(numEdges)
        #flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance + '.pkz')
        flow_after_conservation = pd.read_pickle(out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance +'.pkz')
        
        flow_after_conservation = collections.OrderedDict(sorted(flow_after_conservation.items()))
        
        
        
        with open(out_dir + 'day_comm.txt') as day_file:
            day_ = file.read(day_file)
        
        day_ = int(day_ )
        
        a = []
        #x = []
        A = zload(out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')
        A = np.asmatrix(A)
        P = zload(out_dir + 'OD_pair_route_incidence_'+ instance + files_ID + '.pkz')
        P = np.asmatrix(P)   
        
        for day_ in week_day_Apr_list:    
            x = np.zeros(numEdges)
            L = np.size(P, 0)        
            for ts in flow_after_conservation : 
                #ts = flow_after_conservation.keys()[0]
                #x = np.zeros(numEdges)
                day = (ts.astype('datetime64[D]') - ts.astype('datetime64[M]') + 1).astype(int)
                if day == day_ :
                    
                    a = np.array(list(flow_after_conservation[ts].values()))
                    x = np.c_[x,a]
                    x_ins = np.c_[x_ins,a]
            
            x = np.delete(x,0,1)
            x = np.asmatrix(x)
            
    
            
            x = np.nan_to_num(x)
            y = np.array(np.transpose(x))
            y = y[np.all(y != 0, axis=1)]
            x = np.transpose(y)
            x = np.matrix(x)
            
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
            
            #xi_list = np.asarray(xi_list)
            
            while lam_list == None:
                try:
                    len_x = np.size(x,1)
                    sample_size = np.random.randint(.5*len_x ,len_x)
                    col_idx = np.random.choice(range(len_x), sample_size, replace=False)
                    x1 = x[:,col_idx]
                    cnt_ += 1
                    print(cnt_)
                    if cnt_ >= 45:
                        lam_list = 1
                        print('error, no PSD Q was found')
                    lam_list = GLS2(x1, A, P, L)
            
                except:
                    pass 
                
            saveDemandVec(numNodes, out_dir, instance, files_ID, lam_list, month_w, str(day_) )


        # Calculating the aggregated OD Demand for each instance, used to estimate cost function coefficients
        
        x_ins = np.delete(x_ins,0,1)
        x_ins = np.asmatrix(x_ins)
        

        
        x_ins = np.nan_to_num(x_ins)
        y = np.array(np.transpose(x_ins))
        y = y[np.all(y != 0, axis=1)]
        x_ins = np.transpose(y)
        x_ins = np.matrix(x_ins)

        xi_list = None
        try:
            xi_list = GLS(x_ins, A, L)
        except:
            pass
           
        cnt_ = 0
        while xi_list == None:
            try:
                len_x = np.size(x_ins,1)
                sample_size = np.random.randint(.5*len_x ,len_x)
                col_idx = np.random.choice(range(len_x), sample_size, replace=False)
                x1 = x_ins[:,col_idx]
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
            lam_list = GLS2(x_ins, A, P, L)
        except:
            pass
        
        cnt_ = 0
        
        
        while lam_list == None:
            try:
                len_x = np.size(x_ins,1)
                sample_size = np.random.randint(.5*len_x ,len_x)
                col_idx = np.random.choice(range(len_x), sample_size, replace=False)
                x1 = x_ins[:,col_idx]
                cnt_ += 1
                print(cnt_)
                if cnt_ >= 45:
                    lam_list = 1
                    print('error, no PSD Q was found')
                lam_list = GLS2(x1, A, P, L)
        
            except:
                pass 

        saveDemandVec(numNodes, out_dir, instance, files_ID, lam_list, month_w, 'full' )
