from utils import *
def create_node_link_incidence(instance):
    from utils import *
    import json
    link_label_dict = zload( out_dir + 'link_label_dict.pkz')
    
    instance = time_instances['id'][0]

    with open(out_dir + "/data_traffic_assignment_uni-class/" + files_ID + '_net_' + month_w + '_full_' + instance + '.txt') as MA_journal_flow:
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
    
    N = np.zeros((numNodes, numLinks))
    
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

execfile('../parameters.py')
for instance in time_instances['id']:
	create_node_link_incidence(instance)
