def get_observed_flow(out_dir, files_ID, instance, month_w,  month, year,  week_day_Apr_list):
    from utils import *
    import numpy as np
    import json
    import collections

    link_edge_dict = zload( out_dir + 'link_edge_dict' + files_ID + '.pkz' )

    with open(out_dir + 'link_label_dict_' + '.json', 'r') as json_file:
            link_label_dict_ = json.load(json_file)

    numEdges = len(link_label_dict_)

    with open(out_dir + 'link_min_dict'+ files_ID + '.json', 'r') as json_file:
            link_day_minute_Apr_dict_JSON = json.load(json_file)

    numDays = len(week_day_Apr_list)

    flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance + '.pkz')
    #flow_after_conservation = pd.read_pickle(out_dir + 'flows_before_QP_2_' + files_ID + '_' + instance + '.pkz')

    flow_after_conservation = collections.OrderedDict(sorted(flow_after_conservation.items()))

    link_day_Apr_list = []
    link_vector = []
    for edge in link_edge_dict.values():
  
        for day in week_day_Apr_list: 
            key = 'link_' +  str(edge) + '_' + str(year) + '_' + str(month) + '_'   + str(day) 
            link_day_Apr_list.append(link_day_minute_Apr_dict_JSON[key] ['avg_flow_' + instance ])
        
        link_vector.append(edge)

    x_ = np.matrix(link_day_Apr_list)
    x_ = np.matrix.reshape(x_, numEdges, numDays)
    x_ = np.nan_to_num(x_)
    
    y = [np.mean(x_, axis=1)[i,0] for i in range(x_.shape[0])]

    with open(out_dir +  'link_flow_observ_' + month_w + '_'+ instance + '.json', 'w') as json_file:
        json.dump(y, json_file)
        
    return x_, y, link_vector