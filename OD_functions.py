#### ------------ OD Estimation ------------- :
from utils import *
from math import exp
from functions import *
from numpy.linalg import inv

def od_pair_definition(out_dir, files_ID ):
    G = zload(out_dir + 'G' + files_ID + '.pkz')
    od_pairs=[]
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                od_pairs.append((i, j))
    zdump(od_pairs, out_dir + 'od_pairs'+ files_ID + '.pkz')
    
    with open(out_dir + 'od_pairs'+ files_ID + '.txt', 'w') as f:
        cnt = 0
        for od in od_pairs:
            f.write(str(cnt) + "\t" + str(od) + '\n')
            cnt += 1

def routes(G, out_dir, files_ID, od_pairs, number_of_routes_per_od, instance):
    # Create Routes
    routes_ = []
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
            routes_.append([i[0], i[1], cnt_route])
            OD_pair_route_list.append(cnt_route)
            cnt_route += 1
        OD_pair_route_dict[cnt_od] = OD_pair_route_list
        cnt_od += 1
            
        # Create Path-Link Incidence Matrix
        r = len(routes_) # Number of routes
        m = len(list(G.edges())) # Number of links
        A = np.zeros((m, r))
        for route2 in routes_:
            edgesinpath=zip(route2[0][0:],route2[0][1:])           
            for edge in edgesinpath:
                #print(link)
                link = link_dict[edge] 
                A[link,routes_.index(route2)] = 1
                
    zdump(A, out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')

    #length_of_route_list = [[i[1][0], i[2]] for i in routes]
    length_of_route_dict = {}
    for i in routes_:
        length_of_route_dict[i[2]]=i[1][0] 
        
    # calculate route choice probability matrix P
    # logit choice parameter
    theta = 0.8 #send as parameter !
    s = len(od_pairs) # number of OD pairs
    r = len(routes_) # Number of routes
    P = np.zeros((s, r))
    for i in range(s):
        for r_ in OD_pair_route_dict[i]:
            P[i, r_] = 1
            
            P[i, r_] = exp(- theta * length_of_route_dict[r_]) / \
            sum([exp(- theta * length_of_route_dict[j]) \
                             for j in OD_pair_route_dict[i]])
    
    
    zdump(P, out_dir + 'OD_pair_route_incidence_'+ instance + files_ID + '.pkz')     
    zdump(routes_, out_dir + 'routes_info'+ instance + '.pkz')
    #return routes, A, P

#number_of_routes_per_od = 3
#routes = routes(G, od_pairs, number_of_routes_per_od)

def filter_routes_a(out_dir, instance, files_ID, lower_bound_route):
    A_ = zload(out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')
    P_ = zload(out_dir + 'OD_pair_route_incidence_'+ instance + files_ID + '.pkz')
   # print(np.size(P_,1))
    routes_ =  zload(out_dir + 'routes_info'+ instance + '.pkz')
    routes_ = np.c_[np.sum(P_,axis=0), routes_]
    routes_ = pd.DataFrame(routes_)
    routes_ = routes_[routes_[0] > lower_bound_route] 
    P_t = np.transpose(P_)
    A_t = np.transpose(A_)  
    idx = np.where(sum(P_,0)>lower_bound_route)
    #print(np.size(idx,1))
    #print(np.size(P_t,0))
    P_t = P_t[idx]
    #print(np.size(P_t,0))
    P = np.transpose(P_t)
    #print(np.size(P,1))
    A_t = A_t[idx]
    A = np.transpose(A_t)
    P = [[float(i)/sum(P[j]) for i in P[j]] for j in range(len(P))]
    #P2 = np.zeros((len(P), len(P[0])))
    #for i in range(len(P)):
    #    for j in range(len(P[i])):
    #        P2[i,j] = P[i][j]
    P2 = P
    #print(np.size(P2,1))
    np.save(out_dir + 'path-link_incidence_matrix_filt'+ instance + files_ID , A)
    #zdump(A, out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')
    np.save(out_dir + 'OD_pair_route_incidence_filt_'+ instance + files_ID , P2) 
    #zdump(P_t, out_dir + 'OD_pair_route_incidence_filt_'+ instance + files_ID + '.pkz') 
    np.save(out_dir + 'routes_info_filt_'+ instance , routes_)    
    #zdump(routes_, out_dir + 'routes_info_filt_'+ instance + '.pkz')
    np.savetxt(out_dir + 'path-link_incidence_matrix_filt' + instance + files_ID + '.txt', A, '%i')
    


def path_incidence_matrix(out_dir, files_ID, time_instances, number_of_routes_per_od, theta, lower_bound_route ):
    G_ = zload( out_dir + 'G_' + files_ID + '.pkz' )
    od_pairs = zload(out_dir + 'od_pairs'+ files_ID + '.pkz')
    
    for instance in list(time_instances['id']):
        G = G_[instance]
        routes(G, out_dir, files_ID, od_pairs, number_of_routes_per_od, instance)
    
    for instance in list(time_instances['id']):
        filter_routes_a(out_dir, instance, files_ID, lower_bound_route)
        

def path_incidence_matrix_jing(out_dir, files_ID, time_instances, month_id, number_of_routes_per_od, theta, lower_bound_route ):
   
    import json
    from collections import defaultdict
    import networkx as nx

    for instance in time_instances['id']:
        with open(out_dir + 'data_traffic_assignment_uni-class/'+ files_ID +  '_net_' + month_id + '_full_' + instance + '.txt') as MA_journal_flow:
            MA_journal_flow_lines = MA_journal_flow.readlines()
            MA_journal_links = []
            i = -9
            for line in MA_journal_flow_lines:
                i += 1
                if i > 0:
                    MA_journal_links.append(line.split('  ')[1:3])
            numLinks = i

        link_list_js = [str(int(MA_journal_links[i][0])) + ',' + str(int(MA_journal_links[i][1])) for \
             i in range(len(MA_journal_links))]

        link_list_pk = [str(int(MA_journal_links[i][0])) + '->' + str(int(MA_journal_links[i][1])) for \
                     i in range(len(MA_journal_links))]

        zdump(link_list_js, out_dir + 'link_list_js' + files_ID + '.pkz')

        zdump(link_list_pk, out_dir + 'link_list_pk' + files_ID + '.pkz')

        numNodes = max([int(MA_journal_links[i][1]) for i in range(numLinks)])

        node_neighbors_dict = defaultdict(list)

        for node in range(numNodes):
            for link in MA_journal_links:
                if node == int(link[0]):
                    node_neighbors_dict[str(node)].append(int(link[1]))

        with open(out_dir + 'data_traffic_assignment_uni-class/'+ files_ID +  '_trips_' + month_id + '_full_' + instance + '.txt') as MA_journal_trips:
            MA_journal_trips_lines = MA_journal_trips.readlines()

        numZones = int(MA_journal_trips_lines[0].split(' ')[3])

        od_pairs = []
        for i in range(numZones+1)[1:]:
            for j in range(numZones+1)[1:]:
                if i != j:
                    od_pairs.append([i, j])

        # create O-D pair labels
        # create a dictionary mapping O-D pairs to labels

        OD_pair_label_dict = {}
        OD_pair_label_dict_ = {}

        label = 1
        for i in range(numZones + 1)[1:]:
            for j in range(numZones + 1)[1:]:
                key = (i, j)
                OD_pair_label_dict[str(key)] = label
                OD_pair_label_dict_[str(label)] = key
                label += 1
                
        with open( out_dir + 'od_pair_label_dict.json', 'w') as json_file:
            json.dump(OD_pair_label_dict, json_file)
            
        with open( out_dir + 'od_pair_label_dict__.json', 'w') as json_file:
            json.dump(OD_pair_label_dict_, json_file)


        OD_pair_label_dict_refined = {}
        OD_pair_label_dict_refined_ = {}

        label = 1
        for i in range(numZones + 1)[1:]:
            for j in range(numZones + 1)[1:]:
                if i != j:
                    key = (i, j)
                    OD_pair_label_dict_refined[str(key)] = label
                    OD_pair_label_dict_refined_[str(label)] = key
                    label += 1
                
        with open( out_dir + 'od_pair_label_dict_MA_refined.json', 'w') as json_file:
            json.dump(OD_pair_label_dict_refined, json_file)
            
        with open( out_dir + 'od_pair_label_dict__refined.json', 'w') as json_file:
            json.dump(OD_pair_label_dict_refined_, json_file)
            
            
        # create link labels
        # create a dictionary mapping directed links to labels
        link_label_dict = {}
        link_label_dict_ = {}

        link_list = zload(out_dir + 'link_list_js' + files_ID + '.pkz')

        for i in range(numLinks):
            link_label_dict[str(i)] = link_list[i]

        for i in range(numLinks):
            link_label_dict_[link_list[i]] = i

        with open(out_dir + 'link_label_dict.json', 'w') as json_file:
            json.dump(link_label_dict, json_file)
            
        with open( out_dir + 'link_label_dict_.json', 'w') as json_file:
            json.dump(link_label_dict_, json_file)
            
        # create link labels
        # create a dictionary mapping directed links to labels
        link_label_dict = {}
        link_label_dict_ = {}

        link_list = zload(out_dir + 'link_list_pk' + files_ID + '.pkz')

        for i in range(numLinks):
            link_label_dict[str(i)] = link_list[i]

        for i in range(numLinks):
            link_label_dict_[link_list[i]] = i

        zdump(link_label_dict, out_dir + 'link_label_dict.pkz')
        zdump(link_label_dict_, out_dir + 'link_label_dict_.pkz')

        link_length_dict_MA_journal = {}  # save free-flow time actually
        link_capac_dict_MA_journal = {}

        length_list = []
        capac_list = []

        with open(out_dir + 'data_traffic_assignment_uni-class/'+ files_ID +  '_net_' + month_id + '_full_' + instance + '.txt', 'r') as f:
            read_data = f.readlines()
            flag = 0
            for row in read_data:
                if ';' in row:
                    flag += 1
                    if flag > 1:
                        length_list.append(float(row.split('  ')[5]))
                        capac_list.append(float(row.split('  ')[3]))
                        
        for idx in range(len(length_list)):
            key = str(idx)
            link_length_dict_MA_journal[key] = length_list[idx]
            link_capac_dict_MA_journal[key] = capac_list[idx]

        with open( out_dir + 'link_length_dict.json', 'w') as json_file:
            json.dump(link_length_dict_MA_journal, json_file)
            
        with open( out_dir + 'link_capac_dict.json', 'w') as json_file:
            json.dump(link_capac_dict_MA_journal, json_file)

            # compute length of a route
        def routeLength(route):
            link_list = []
            node_list = []
            for i in route.split('->'):
                node_list.append(int(i))
            for i in range(len(node_list))[:-1]:
                link_list.append('%d->%d' %(node_list[i], node_list[i+1]))
            length_of_route = sum([link_length_dict_MA_journal[str(link_label_dict_[link])] for link in link_list])
            return length_of_route


        MA_journal = nx.DiGraph()

        MA_journal.add_nodes_from(range(numNodes+1)[1:])

        MA_journal_weighted_edges = [(int(link_list_js[i].split(',')[0]), int(link_list_js[i].split(',')[1]), \
                                   length_list[i]) for i in range(len(link_list_js))]

        MA_journal.add_weighted_edges_from(MA_journal_weighted_edges)

        path = nx.all_pairs_dijkstra_path(MA_journal)
        path = list(path)
        #print(path)
        with open(out_dir + 'path-link_incidence_' + instance + files_ID + '.txt', 'w') as the_file:
            for od in od_pairs:
                origi = od[0]
                desti = od[1]
                the_file.write('O-D pair (%s, %s):\n'%(origi, desti))
                route = str(path[origi-1][1][desti]).replace("[", "").replace(", ", "->").replace("]", "")
                the_file.write(route)
                the_file.write('\n')

        with open(out_dir + 'path-link_incidence_' + instance + files_ID + '.txt', 'r') as the_file:
            # path counts
            i = 0  
            for row in the_file:
                if '->' in row:
                    i = i + 1

        with open( out_dir  + 'numRoutes_' + instance + files_ID + '.json', 'w') as json_file:
            json.dump(i, json_file)



def GLS(x, A):
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

    L = len(Q)

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
    gls_cost =model.objVal
    return xi_list, gls_cost

'''
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
 
    return xi_list, model.objVal

'''
def GLS2(x, A, P, L):
    import numpy as np
    from numpy.linalg import inv
    import json    
    """
    x: sample matrix, each column is a link flow vector sample; 24 * K
    A: path-link incidence matrix
    P: logit route choice probability matrix
    L: dimension of lam
    ----------------
    return: lam
    ----------------
    """
    K = np.size(x, 1)
    S = samp_cov(x)
    inv_S = inv(S).real

    A_t = np.transpose(A)
    P_t = np.transpose(P)
    # PA'
    PA_t = np.dot(P, A_t)
    # AP_t
    AP_t = np.transpose(PA_t)

    Q = np.dot(np.dot(PA_t, inv_S), AP_t)
    b = sum([np.dot(np.dot(PA_t, inv_S), x[:, k]) for k in range(K)])

    model = Model("OD_matrix_estimation")

    lam = []
    for l in range(L):
        lam.append(model.addVar(name='lam_' + str(l)))

    model.update() 

    # Set objective: (K/2) lam' * Q * lam - b' * lam
    obj = 0
    for i in range(L):
        for j in range(L):
            obj += (1.0 /2) * K * lam[i] * Q[i, j] * lam[j]
    for l in range(L):
        obj += - b[l] * lam[l]
    model.setObjective(obj)

    # Add constraint: lam >= 0
    for l in range(L):
        model.addConstr(lam[l] >= 0)

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
    for l in range(size(P,0)):
        lam.append(model.addVar(name='lam_' + str(l)))
        model.update()
        model.addConstr(lam[l] >= 0)
        
    p = {}
    for i in range(size(P,0)):
        for j in range(np.size(P,1)):
            p[(i,j)] = model.addVar(name='p_' + str(i) + ',' + str(j))  
            model.update()
            model.addConstr(p[(i,j)] >= 0)
            if P[i,j] == 0:
                model.addConstr(p[(i,j)] == 0)
    
    for i in range(size(P,0)):
        model.addConstr(quicksum(p[(i,j)] for j in range(size(P,1))) == 1)
    
    for idx in range(len(xi_list)):
        model.addConstr(quicksum(p[(l,idx)] * lam[l] for l in range(L)) >= xi_list[idx])
        model.addConstr(quicksum(p[(l,idx)] * lam[l] for l in range(L)) <= xi_list[idx])
    
    model.update()
    
    obj = 0
    model.setObjective(quicksum(p[1,j] for j in range(size(P,1))))
    
    model.update() 
    
    model.setParam('OutputFlag', False)
    model.optimize()
    
    lam_list = []
    for v in model.getVars():
        # print('%s %g' % (v.varName, v.x))
        if 'lam' in v.varName:
            lam_list.append(v.x)
            
    return lam_list

'''

def GLSp2(xi_list, P, L):
    
    P[P>0]=1
    mGLSJulia = Model('GLSp2')
    
    lam = []
    
    for i in range(size(P,0)):
        lam.append(mGLSJulia.addVar(name = 'lam_'+ str(i)))
        mGLSJulia.addNLConstraint(quicksum(p[i,j] for j in range(size(P,2))) == 1)
        
        for j in range(size(P,1)):
            p[(i,j)] = mGLSJulia.addVar(name='p_' + str(i) + ',' + str(j))
            
            if P[i,j] == 0:
                mGLSJulia.addConstr(p[(i,j)] >= 0)
    
    mGLSJulia.update()
    
    for i in range(size(P,0)):
        mGLSJulia.addNLConstraint(sum{p[i,j], j = 1:size(P,1)} == 1)
        
    for l in range(len(xi_list)):
        mGLSJulia.addNLConstraint(sum{p[i,l] * lam[i], i = 1:size(P,1)} == xi_list[l])
    
    mGLSJulia.update()
    
    mGLSJulia.setNLObjective(Min, sum{p[1,j], j = 1:size(P,2)})  # play no actual role, but could not use zero objective
    
    mGLSJulia.update()

    model.setParam('OutputFlag', False)
    model.optimize()
    
'''
        
def runGLS(out_dir, files_ID, time_instances, month_id):
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
            xi_list, gls_cost = GLS(x, A, L)
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
        while lam_list == None:
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
        

        def saveDemandVec(G, out_dir, instance, files_ID, lam_list, month_id ):
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
                            
        saveDemandVec(G, out_dir, instance, files_ID, lam_list, month_id )
      