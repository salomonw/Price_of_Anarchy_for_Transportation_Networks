
#### ------------ OD Estimation ------------- :
from utils import *
from math import exp
from functions import *

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

def filter_routes(out_dir, instance, files_ID, lower_bound_route):
    A = zload(out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')
    P = zload(out_dir + 'OD_pair_route_incidence_'+ instance + files_ID + '.pkz')
    P_t = np.transpose(P)
    A_t = np.transpose(A)
    
    idx = np.where(sum(P)>lower_bound_route)
    
    P_t = P_t[idx]
    P = np.transpose(P_t)
    
    A_t = A_t[idx]
    A = np.transpose(A_t)
    
    P = [[float(i)/sum(P[j]) for i in P[j]] for j in range(len(P))]
    P = np.asmatrix(P)
   
    zdump(A, out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')
    zdump(P, out_dir + 'OD_pair_route_incidence_'+ instance + files_ID + '.pkz') 
    
    
def path_incidence_matrix(out_dir, files_ID, time_instances, number_of_routes_per_od, theta ):
    G_ = zload( out_dir + 'G_' + files_ID + '.pkz' )
    od_pairs = zload(out_dir + 'od_pairs'+ files_ID + '.pkz')
    
    for instance in list(time_instances['id']):
        G = G_[instance]
        routes(G, out_dir, files_ID, od_pairs, number_of_routes_per_od, instance)
        filter_routes(out_dir, instance, files_ID, lower_bound_route)
        
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


def GLS2(x, A, P, L):
    """
    x: sample matrix, each column is a link flow vector sample; 24 * K
    A: path-link incidence matrix
    P: logit route choice probability matrix
    L: dimension of lam
    ----------------
    return: lam
    ----------------
    """
    import numpy as np
    from numpy.linalg import inv
    import json
    K = np.size(x, 1)
    S = samp_cov(x)

    #print("rank of S is: \n")
    #print(matrix_rank(S))
    #print("sizes of S are: \n")
    #print(np.size(S, 0))
    #print(np.size(S, 1))

    inv_S = inv(S).real
    inv_S = inv(S)
    
    A_t = np.transpose(A)
    P_t = np.transpose(P)
    # PA'
    PA_t = np.dot(P, A_t)

    #print("rank of PA_t is: \n")
    #print(matrix_rank(PA_t))
    #print("sizes of PA_t are: \n")
    #print(np.size(PA_t, 0))
    #print(np.size(PA_t, 1))

    # AP_t
    AP_t = np.transpose(PA_t)

    Q_ = np.dot(np.dot(PA_t, inv_S), AP_t)
    Q = adj_PSD(Q_).real  # Ensure Q to be PSD
    Q = Q_
    
    #isPSD(Q)
    
    #print("rank of Q is: \n")
    #print(matrix_rank(Q))
    #print("sizes of Q are: \n")
    #print(np.size(Q, 0))
    #print(np.size(Q, 1))

    b = sum([np.dot(np.dot(PA_t, inv_S), x[:, k]) for k in range(K)])
    # print(b[0])
    # assert(1==2)

    model = Model("OD_matrix_estimation")

    lam = []
    for l in range(L):
        lam.append(model.addVar(name='lam_' + str(l)))

    model.update() 

    # Set objective: (K/2) lam' * Q * lam - b' * lam
    obj = 0
    for i in range(L):
        for j in range(L):
            obj += (1.0 / 2) * K * lam[i] * Q[i, j] * lam[j]
    for l in range(L):
        obj += - b[l] * lam[l]
    model.setObjective(obj)

    # Add constraint: lam >= 0
    for l in range(L):
        model.addConstr(lam[l] >= 0)
      #  model.addConstr(lam[l] <= 5000)
    #fictitious_OD_list = zload('../temp_files/fictitious_OD_list')
    #for l in fictitious_OD_list:
        #model.addConstr(lam[l] == 0)
    model.update() 

    model.setParam('OutputFlag', False)
    model.optimize()

    lam_list = []
    for v in model.getVars():
        # print('%s %g' % (v.varName, v.x))
        lam_list.append(v.x)
    # print('Obj: %g' % obj.getValue())
    return lam_list



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
    
    obj = 0
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
    
    #week_day_Apr_list = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]
    week_day_Apr_list = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20]
    #week_day_Apr_list = [2, 25,  30]
    
    for instance in list(time_instances['id']):
        #instance = 'AM'
        flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
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
        
        super_threshold_indices = P > 0.005
        P5[super_threshold_indices] = 1
        
        #lam_list = GLS2(x, A, P, L)

        
        '''
        lam_list = None
        while lam_list is None:
            try:
                len_x = np.size(x,1)
                sample_size = np.random.randint(.5*len_x ,len_x)
                col_idx = np.random.choice(range(len_x), sample_size, replace=False)
                x1 = x[:,col_idx]
                lam_list = GLS2(x1, A, P, L)
                print(a)
            except:
                 pass
        '''
        
        
            
                     
        xi_list = GLS(x, A, L)
        
        #lam_list = GLS2(x, A, P, L)
        
        #lam_list = ODandRouteChoiceMat(P, xi_list)
        
        
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
        saveDemandVec(G, out_dir, instance, files_ID, lam_list)
       
    
    
    
    
    
 
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



# include measurement error described in Hazelton paper














































