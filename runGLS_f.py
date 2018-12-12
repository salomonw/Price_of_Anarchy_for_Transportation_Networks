from parameters import *
from utils_julia import *
from OD_functions import *
#from functions import *
import numpy as np
from numpy.linalg import inv
import json
import collections 
import math

# runGLS_f exectuable from Julia (delete writing the txt files)
def GLS_p1(instance, day_, average_over_time):
    try:
        del A,P
    except:
        []
    time_window = average_over_time
    edges = zload(out_dir + 'link_length_dict.pkz')
    node_link_inc = zload(out_dir + 'node_link_incidence.pkz')    
    numEdges = len(edges)
    numNodes = len(node_link_inc)


    week_day_Apr_list = week_day_list

    #for instance in time_instances['id']:
    gls_cost_ = {}
    x_ins = np.zeros(numEdges)
    #flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance + '.pkz')
    flow_after_conservation = pd.read_pickle(out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance +'.pkz')
    #flow_after_conservation = pd.read_pickle(out_dir + 'density_links' + files_ID + '_' + instance +'.pkz')
    #flow_after_conservation = pd.read_pickle(out_dir + 'density_links_before_QP' + files_ID + '_' + instance +'.pkz')

    flow_after_conservation = collections.OrderedDict(sorted(flow_after_conservation.items()))

    a = []
    #x = []
    A = np.load(out_dir + 'path-link_incidence_matrix_filt'+ instance + files_ID + '.npy')
   # A = np.asmatrix(A)
    P = np.load(out_dir + 'OD_pair_route_incidence_filt_'+ instance + files_ID + '.npy')
   # P = np.asmatrix(P)   

    #for day_ in week_day_Apr_list:    
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

    x = np.delete(x_ins,0,1)
    x = np.asmatrix(x)



    x = np.nan_to_num(x)
    y = np.array(np.transpose(x))
    y = y[np.all(y != 0, axis=1)]
    x = np.transpose(y)
    x = np.matrix(x)

    xi_list = None
    try:
        xi_list, gls_cost = GLS(x, A)
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
            xi_list, gls_cost = GLS(x1, A)

        except:
            pass

    return xi_list, x, A, P, gls_cost


# runGLS_f exectuable from Julia, for flows over days
def GLS_p1_all(instance, week_day_list, average_over_time, period): # notice that period can be "all" or "daily_avg"
    time_window = average_over_time
    edges = zload(out_dir + 'link_length_dict.pkz')
    node_link_inc = zload(out_dir + 'node_link_incidence.pkz')    
    numEdges = len(edges)
    numNodes = len(node_link_inc)
    x2 = np.zeros(numEdges)
    for day_ in week_day_list:
        week_day_Apr_list = week_day_list
        #for instance in time_instances['id']:
        gls_cost_ = {}
        x_ins = np.zeros(numEdges)
        #flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance + '.pkz')
        flow_after_conservation = pd.read_pickle(out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance +'.pkz')
        #flow_after_conservation = pd.read_pickle(out_dir + 'density_links' + files_ID + '_' + instance +'.pkz')
        #flow_after_conservation = pd.read_pickle(out_dir + 'density_links_before_QP' + files_ID + '_' + instance +'.pkz')
        flow_after_conservation = collections.OrderedDict(sorted(flow_after_conservation.items()))
        a = []
        A = np.load(out_dir + 'path-link_incidence_matrix_filt'+ instance + files_ID + '.npy')
        #A = np.asmatrix(A)
        P = np.load(out_dir + 'OD_pair_route_incidence_filt_'+ instance + files_ID + '.npy')
        #P = np.asmatrix(P)   
       
        #for day_ in week_day_Apr_list:    
        x = np.zeros(numEdges)
        L = np.size(P, 0)        
        for ts in flow_after_conservation : 
            #ts = flow_after_conservation.keys()[0]
            x = np.zeros(numEdges)
            day = (ts.astype('datetime64[D]') - ts.astype('datetime64[M]') + 1).astype(int)
            if day == day_ :
                
                a = np.array(list(flow_after_conservation[ts].values()))
                x = np.c_[x,a]
                
                x_ins = np.c_[x_ins,a]

        x = np.delete(x_ins,0,1)
        x = np.asmatrix(x)

        x = np.nan_to_num(x)
        y = np.array(np.transpose(x))
        y = y[np.all(y != 0, axis=1)]
        x = np.transpose(y)
        x = np.matrix(x)

        if period == "all":
            x2 = np.c_[x2, x]
          #  x2 = np.delete(x2,0,1)
        if period == "daily_avg":
            x =  np.mean(x, axis=1)
            x = np.matrix(x)
            x2 = np.c_[x2 ,x]
          #  x2 = np.delete(x2,0,1)
    x2 = np.delete(x2,0,1)
    xi_list = None
    try:
        xi_list, gls_cost = GLS(x2, A)
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
            xi_list, gls_cost = GLS(x1, A)
            return xi_list, x, A, P, gls_cost
        except:
            pass
    return xi_list, x, A, P, gls_cost














'''
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

        gls_cost_[day_] = gls_cost

        
    # Calculating the aggregated OD Demand for each instance, used to estimate cost function coefficients
    

    x_ins = np.delete(x_ins,0,1)
    x_ins = np.asmatrix(x_ins)
    

    with open(out_dir + 'OD_demands/gls_cost_vec_'+ month_w + '_weekday_'+ instance + files_ID + '.json', 'w') as json_file:
        json.dump(gls_cost_, json_file)
    
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
            xi_list, gls_cost = GLS(x1, A, L)
    
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
'''