# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 16:51:24 2018

@author: Salomon Wollenstein
"""

from utils import *
import json
import numpy as np
import matplotlib.pyplot as plt
import pylab
from pylab import *
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA

def parse_data_for_TAP(out_dir, files_ID, time_instances, month_w):

    # ------------------ COMPUTE JACOBIAN --------------------------
    
    for instance in list(time_instances['id']):
        with open(out_dir + "/data_traffic_assignment_uni-class/" + files_ID + '_net_' + month_w + '_' + instance + '.txt') as MA_journal_flow:
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
        
        numNodes = max([int(MA_journal_links[i][1]) for i in range(numLinks)])
        
        from collections import defaultdict
        
        node_neighbors_dict = defaultdict(list)
        
        for node in range(numNodes):
            for link in MA_journal_links:
                if node == int(link[0]):
                    node_neighbors_dict[str(node)].append(int(link[1]))
        
        with open(out_dir + "/data_traffic_assignment_uni-class/" + files_ID + '_trips_' + month_w + '_' + instance + '.txt') as MA_journal_trips:
            MA_journal_trips_lines = MA_journal_trips.readlines()
    
    numZones = int(MA_journal_trips_lines[0].split(' ')[3])
    
    od_pairs = []
    for i in range(numZones+1)[1:]:
        for j in range(numZones+1)[1:]:
            if i != j:
                od_pairs.append([i, j])
                
    numODpairs = len(od_pairs)
    
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
            
    with open(out_dir + 'od_pair_label_dict.json', 'w') as json_file:
        json.dump(OD_pair_label_dict, json_file)
        
    with open(out_dir + 'od_pair_label_dict.json', 'w') as json_file:
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
            
    with open(out_dir + 'od_pair_label_dict_refined.json', 'w') as json_file:
        json.dump(OD_pair_label_dict_refined, json_file)
        
    with open(out_dir + 'od_pair_label_dict__refined.json', 'w') as json_file:
        json.dump(OD_pair_label_dict_refined_, json_file)
        
        
    # create link labels
    # create a dictionary mapping directed links to labels
    link_label_dict = {}
    link_label_dict_ = {}
    
    for i in range(numLinks):
        link_label_dict[str(i)] = link_list_js[i]
    
    for i in range(numLinks):
        link_label_dict_[link_list_js[i]] = i
    
    with open(out_dir + 'link_label_dict.json', 'w') as json_file:
        json.dump(link_label_dict, json_file)
        
    with open(out_dir + 'link_label_dict_.json', 'w') as json_file:
        json.dump(link_label_dict_, json_file)
        
    # create link labels
    # create a dictionary mapping directed links to labels
    link_label_dict = {}
    link_label_dict_ = {}
    
    for i in range(numLinks):
        link_label_dict[str(i)] = link_list_pk[i]
    
    for i in range(numLinks):
        link_label_dict_[link_list_pk[i]] = i
    
    zdump(link_label_dict, out_dir + 'link_label_dict_network.pkz')
    zdump(link_label_dict_, out_dir +  'link_label_dict_network_.pkz')
    
    link_length_list = []
    with open( out_dir + 'data_traffic_assignment_uni-class/'+ files_ID + '_net_' + month_w + '_' + instance + '.txt', 'r') as f:
        read_data = f.readlines()
        flag = 0
        for row in read_data:
            if ';' in row:
                flag += 1
                if flag > 1:
                    link_length_list.append(float(row.split('  ')[5]))
    
    link_label_dict = zload( out_dir + 'link_label_dict_network.pkz')
    link_label_dict_ = zload( out_dir + 'link_label_dict_network_.pkz')
    
    #import networkx as nx
    
    def jacobianSpiess(numNodes, numLinks, numODpairs, od_pairs, link_list_js, link_length_list):
        MA_journal = nx.DiGraph()
    
        MA_journal.add_nodes_from(range(numNodes+1)[1:])
    
        MA_journal_weighted_edges = [(int(link_list_js[i].split(',')[0]), int(link_list_js[i].split(',')[1]), \
                                 link_length_list[i]) for i in range(len(link_list_js))]
    
        MA_journal.add_weighted_edges_from(MA_journal_weighted_edges)
    
        path = list(nx.all_pairs_dijkstra_path(MA_journal))
    
        od_route_dict = {}
        for od in od_pairs:
            origi = od[0]
            desti = od[1]
            key = OD_pair_label_dict_refined[str((origi, desti))]
            route = str(path[origi-1][1][desti]).replace("[", "").replace(", ", "->").replace("]", "")
            od_route_dict[key] = route
    
        od_link_dict = {}
        for idx in range(len(od_route_dict)):
            od_link_list = []
            od_node_list = od_route_dict[idx+1].split('->')
            for i in range(len(od_node_list)):
                if i < len(od_node_list) - 1:
                    od_link_list.append(link_label_dict_[od_node_list[i] + '->' + od_node_list[i+1]])
            od_link_dict[idx] = od_link_list
    
        jacob = np.zeros((numODpairs, numLinks))
    
        for i in range(numODpairs):
            for j in range(numLinks):
                if j in od_link_dict[i]:
                    jacob[i, j] = 1
    
        return jacob
    
    jacob = jacobianSpiess(numNodes, numLinks, numODpairs, od_pairs, link_list_js, link_length_list)
    
    # ------------------ CREATE PATH LINK INCIDENCE MATRIX ----------------------------

    for instance in list(time_instances['id']):
    
        P = zload(out_dir + 'OD_pair_route_incidence_'+ instance + files_ID + '.pkz')
        
        P_dict = {}
        
        for i in range(np.shape(P)[0]):
            for j in range(np.shape(P)[1]):
                key = str(i+1) + '-' + str(j+1)
                if (P[i, j] > 1e-1):
                    P_dict[key] = P[i, j]
        
        with open(out_dir + 'od_pair_route_incidence_'+ instance + files_ID +'.json', 'w') as json_file:
            json.dump(P_dict, json_file)
        
        A = zload(out_dir + 'path-link_incidence_matrix_'+ instance + files_ID + '.pkz')
        
        A_dict = {}
        
        for i in range(np.shape(A)[0]):
            for j in range(np.shape(A)[1]):
                key = str(i+1) + '-' + str(j+1)
                if (A[i, j] > 1e-1):
                    A_dict[key] = A[i, j]
                
        with open(out_dir + 'link_route_incidence_'+ instance + files_ID +'.json', 'w') as json_file:
            json.dump(A_dict, json_file)
        
        from numpy.linalg import inv, matrix_rank
        
        A_t = np.transpose(A)
        P_t = np.transpose(P)
        # PA'
        AP_t = np.dot(A, P_t)
        print("rank of P is: ")
        print(matrix_rank(P))
        print("sizes of P are: ")
        print(np.size(P, 0))
        print(np.size(P, 1))
        print("rank of A is: ")
        print(matrix_rank(A))
        print("sizes of A are: ")
        print(np.size(A, 0))
        print(np.size(A, 1))
        print("rank of AP_t is: ")
        print(matrix_rank(AP_t))
        print("shape of AP_t is: ")
        print(np.shape(AP_t))
    
    
    
    #### ---------- CREATE NODE LINK ----------------
    link_label_dict = zload(out_dir + 'link_label_dict_network.pkz')
    N = np.zeros((numZones, numLinks))
    N_dict = {}
    for j in range(np.shape(N)[1]):
        for i in range(np.shape(N)[0]):
            if (str(i+1) == link_label_dict[str(j)].split('->')[0]):
                N[i, j] = 1
            elif (str(i+1) == link_label_dict[str(j)].split('->')[1]):
                N[i, j] = -1
            key = str(i) + '-' + str(j)
            N_dict[key] = N[i, j]
            
    with open(out_dir + 'node_link_incidence.json', 'w') as json_file:
        json.dump(N_dict, json_file)
        
    zdump(N, out_dir + 'node_link_incidence.pkz')



####### -------------- 08_InverseVI_uni_MA_with_base_trans_python -----------------

def InverseVI_uni_MA_with_base_trans_python(out_dir, files_ID, time_instances, month_w):
    
    for instance in list(time_instances['id']):

        N = zload(out_dir + 'node_link_incidence.pkz')
        def g_true(t):
            return 1 + 0.15 * (t ** 4)
        
        def polyEval(coeffs, pt):
            return sum([coeffs[i] * (pt ** i) for i in range(len(coeffs))])
        
        capac_list = []
        free_flow_time_list = []
        capac_dict = {}
        free_flow_time_dict = {}
        
        with open(out_dir + 'data_traffic_assignment_uni-class/'+ files_ID + '_net_' + month_w + '_' + instance + '.txt', 'r') as f:
            read_data = f.readlines()
        
        for row in read_data:
            if len(row.split()) == 11:
                key = row.split()[0] + ',' + row.split()[1]
                capac_list.append(float(row.split()[2]))
                free_flow_time_list.append(float(row.split()[4]))
                capac_dict[key] = float(row.split()[2])
                free_flow_time_dict[key] = float(row.split()[4])
        
        # read in link labels
        with open(out_dir + 'link_label_dict.json', 'r') as json_file:
            link_label_dict = json.load(json_file)
            
        # read in demand data
        with open(out_dir + 'demands' + files_ID + '.json', 'r') as json_file:
            demands = json.load(json_file) 
            
        numNode = N.shape[0]
        numLink = N.shape[1]
        assert(numLink == len(capac_list))
        
        flow_list = []
        flow_dict = {}
        
        with open(out_dir + 'flows_converge_'+ month_w + '_' +  instance +'.txt', 'r') as f:
            read_data = f.readlines()
        
        for row in read_data:
            if len(row.split()) == 3:
                key = row.split()[0] + ',' + row.split()[1]
                flow_list.append(float(row.split()[2]))
                flow_dict[key] = float(row.split()[2])
        #         print(row.split())
        
        flow_normalized = [flow_list[i]/capac_list[i] for i in range(numLink)]
        
        def fitCost(c, d, gama):
            normCoeffs = []
        
            for i in range(d+1):
                normCoeffs.append(sc.comb(d, i, exact=True) * (c ** (d-i)))
        
            od_list = []
            for i in range(numNode + 1)[1:]:
                for j in range(numNode + 1)[1:]:
                    if i != j:
                        key = '(' + str(i) + ',' + str(j) + ')'
                        od_list.append(key)
        
            model = Model("InverseVI")
        
            alpha = []
            for i in range(d+1):
                key = str(i)
                alpha.append(model.addVar(name='alpha_' + key))
        
            epsilon = model.addVar(name='epsilon')
        
            yw = {}
            for od in od_list:
                for i in range(numNode):
                    key = od + str(i)
                    yw[key] = model.addVar(name='yw_' + key)
        
            model.update()
        
            # add dual feasibility constraints
            for od in od_list:
                for a in range(numLink):
                    model.addConstr(yw[od+str(int(link_label_dict[str(a)].split(',')[0])-1)] - 
                                    yw[od+str(int(link_label_dict[str(a)].split(',')[1])-1)] <= 
                                    free_flow_time_list[a] * polyEval(alpha, flow_normalized[a]))        
            model.update()
        
            # add increasing constraints
            myList = flow_normalized
            flow_sorted_idx = sorted(range(len(myList)),key=lambda x:myList[x])
            # model.addConstr(polyEval(alpha, 0) <= polyEval(alpha, flow_normalized[flow_sorted_idx[0]]))
            for i in range(numLink):
                if (i < numLink-1):
                    a_i_1 = flow_sorted_idx[i]
                    a_i_2 = flow_sorted_idx[i+1]
                    model.addConstr(polyEval(alpha, flow_normalized[a_i_1]) <= polyEval(alpha, flow_normalized[a_i_2]))
            model.update()
        
            model.addConstr(epsilon >= 0)
            model.update()
        
            # add primal-dual gap constraint
        
            primal_cost = sum([flow_list[a] * free_flow_time_list[a] * polyEval(alpha, flow_normalized[a]) 
                               for a in range(numLink)])
            dual_cost = sum([demands[od] * (yw[od + str(int(od.split(',')[1].split(')')[0])-1)] - 
                                            yw[od + str(int(od.split(',')[0].split('(')[1])-1)]) 
                             for od in od_list])
            
            ref_cost = sum([flow_list[a] * free_flow_time_list[a] for a in range(numLink)])
        
            model.addConstr(primal_cost - dual_cost <= epsilon * ref_cost)
        #     model.addConstr(dual_cost - primal_cost <= epsilon * ref_cost)
        
            model.update()
        
            # add normalization constraint
            model.addConstr(alpha[0] == 1)
            model.update()
        
            # Set objective
            obj = 0
            obj += sum([alpha[i] * alpha[i] / normCoeffs[i] for i in range(d+1)])
            obj += gama * epsilon
        
            model.setObjective(obj)
        
            model.setParam('OutputFlag', False)
            model.optimize()
            alpha_list = []
            for v in model.getVars():
            #     print('%s %g' % (v.varName, v.x))
                if 'alpha' in v.varName:
                    alpha_list.append(v.x)
            return alpha_list
        
        alpha_list = fitCost(1.5, 5, 1.0)
        
        
        
        xs = linspace(0, 2, 20)
        zs_true = [g_true(t) for t in xs]
        
        def g_est(t):
            return polyEval(alpha_list, t)
        
        zs_est = [g_est(t) for t in xs]
        true, = plt.plot(xs, zs_true, "bo-")
        est, = plt.plot(xs, zs_est, "rs-")
        
        plt.legend([true, est], ["g_true", "g_est"], loc=0)
        plt.xlabel('Scaled Flow')
        plt.ylabel('Scaled Cost')
        pylab.xlim(-0.1, 1.6)
        pylab.ylim(0.9, 2.0)
        grid("on")
        savefig('fittedCostFunc_'+'_' + instance + '_' + month_w +'.eps')

from numpy import arange
def calc_testing_errors(out_dir, files_ID, time_instances, month_w, deg_grid, c_grid, lamb_grid, train_idx):
    testing_set = {}
    for instance in time_instances['id']:
        testing_set_1, testing_set_2, testing_set_3 = zload(out_dir + 'testing_sets_' + month_w +'_' + instance + '.pkz')
        
        testing_sets = {}
        
        testing_sets[1] = testing_set_1
        testing_sets[2] = testing_set_2
        testing_sets[3] = testing_set_3
        
        with open(out_dir + '/uni-class_traffic_assignment_MSA_flows_' + month_w + '_' + instance + '.json', 'r') as json_file:
            xl = json.load(json_file)
        testing_errors_dict = {}

        for deg in deg_grid:
            for c in c_grid:
                for lam in lamb_grid:
                    for idx in train_idx:
                        key_ = "'(" + str(deg) + ', ' + str(c) + ', ' + str(lam) + ', ' + str(idx) + ")'"
                        #if lam == 1e-5:
                        #    key_ = "'(" + str(deg) + ',' + str(c) + ',' + '1.0e-5' + ',' + str(idx) + ")'"
                        key_ = key_[1:-1]
                        testing_errors_dict[key_] = np.mean([LA.norm(np.array(xl[key_]) - \
                                                                     np.array(testing_sets[idx])[:, j]) \
                                                             for j in range(np.size(testing_sets[idx], 1))])
        testing_mean_errors_dict = {}
        
        for deg in deg_grid:
            for c in c_grid:
                for lam in lamb_grid:
                    key_ = {}
                    for idx in train_idx:
                        key_[idx] = "'(" + str(deg) + ', ' + str(c) + ', ' + str(lam) + ', ' + str(idx) + ")'"
                        #if lam == 1e-5:
                        #    key_[idx] = "'(" + str(deg) + ',' + str(c) + ',' + '1.0e-5' + ',' + str(idx) + ")'"
                        key_[idx] = key_[idx][1:-1]
                    key__ = "'(" + str(deg) + ',' + str(c) + ',' + str(lam) + ")'"
                    #if lam == 1e-5:
                    #    key__ = "'(" + str(deg) + ',' + str(c) + ',' + '1.0e-5' + ")'"
                    key__ = key__[1:-1]
                    
                    testing_mean_errors_dict[key__] = np.mean([testing_errors_dict[key_[idx]] for idx in train_idx])
                        
                
        testing_mean_errors_switch_dict = {}
        for _key_ in testing_mean_errors_dict.keys():
            testing_mean_errors_switch_dict[testing_mean_errors_dict[_key_]] = _key_
                    
        best_key = '(8,0.5,10000.0,1)'

        # Writing JSON data
        with open(out_dir + 'cross_validation_best_key_' + month_w + '_' + instance + '.json', 'w') as json_file:
            json.dump(best_key, json_file)
                    
        #print(testing_mean_errors_switch_dict)
                    