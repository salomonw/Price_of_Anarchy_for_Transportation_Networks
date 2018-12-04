
def find_paths(out_dir, files_ID, time_instances, month_id):
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

        node_neighbors_dict = {}

        for node_ in range(numNodes):
            node = node_#+1
            node_neighbors_dict[node] = []
            for link in MA_journal_links:
                if node == int(link[0]):
                    node_neighbors_dict[node].append(int(link[1]))

        with open(out_dir + 'data_traffic_assignment_uni-class/'+ files_ID +  '_trips_' + month_id + '_full_' + instance + '.txt') as MA_journal_trips:
            MA_journal_trips_lines = MA_journal_trips.readlines()

        numZones = int(MA_journal_trips_lines[0].split(' ')[3])
        numLinks = len(link_list_js)
        #od_pairs = []
        #for i in range(numZones+1)[1:]:
        #    for j in range(numZones+1)[1:]:
        #        if i != j:
        #            od_pairs.append([i, j])

        # create O-D pair labels
        # create a dictionary mapping O-D pairs to labels

        OD_pair_label_dict = {}
        OD_pair_label_dict_ = {}

        label = 1
        for od in od_pairs:
            i = od[0]
            j = od[1]
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
        for od in od_pairs:
            i = od[0]
            j = od[1]
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
        
        #path = nx.all_simple_paths(MA_journal)
        #path = list(path)
        #print(path)
        route_path_mat = {}
        od_route_dict = {}
        route_ = 0
        
        with open(out_dir + 'path-link_incidence_' + instance + files_ID + '.txt', 'w') as the_file:
            for od in od_pairs:
                origi = od[0]
                desti = od[1]
                paths = list(nx.shortest_simple_paths(MA_journal,origi,desti))
                the_file.write('O-D pair (%s, %s):\n'%(origi, desti))
                for path in paths:
                   # print(path)
                    route = str(path).replace("[", "").replace(", ", "->").replace("]", "")
                    the_file.write(route)
                    the_file.write('\n')
                    route_ = route_+ 1
                    #print(route_)
                    #route_path_mat = route
                    key = "(" + str(origi) + ", " + str(desti) + ")"
                    #print(OD_pair_label_dict_refined)
                    od_route_dict[str(OD_pair_label_dict_refined[key]) + "-" + str(route_)] = 1
                    #route_path_mat[route_] = np.zeros(numLinks )
                    
                    for link_i in range(len(path)-1):
                        link = str(path[link_i]) + "->" + str(path[link_i+1])
                        link_id = link_label_dict_[link]
                        route_path_mat[str(link_id) + "-" + str(route_)] = 1
                        
        
        #link_route_mat = np.transpose(np.matrix( np.array([route_path_mat[i] for i in route_path_mat.keys()])))
        
        #  print(link_route_mat )
        with open(out_dir + 'path-link_incidence_' + instance + files_ID + '.txt', 'r') as the_file:
            # path counts
            i = 0  
            for row in the_file:
                if '->' in row:
                    i = i + 1
                
        with open( out_dir  + 'numRoutes_' + instance + files_ID + '.json', 'w') as json_file:
            json.dump(i, json_file)
            
        with open(out_dir + "link_route_incidence_" + instance + files_ID + ".json", 'w') as json_file:
            json.dump(route_path_mat, json_file)
            
        with open(out_dir + "od_pair_route_incidence_" + instance + files_ID + ".json", 'w') as json_file:
            json.dump(od_route_dict, json_file)
      
         
find_paths(out_dir, files_ID, time_instances, month_id)