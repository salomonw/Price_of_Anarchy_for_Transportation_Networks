#comparison with jing's flows

### Percentiles
#Jing


#Salo
free_flow_speed_salo = pd.read_pickle(out_dir + 'free_flow_speed_ ' + files_ID + '.pkz')
free_flow_speed_jing = zload('C:/Users/Salomon Wollenstein/Documents/GitHub/InverseVIsTraffic/temp_files/tmc_ref_speed_dict.pkz')
### Capacity and network characteristics

#results for jing net
import os 
#os.chdir('C:/Users/Salomon Wollenstein/Documents/GitHub/InverseVIsTraffic/Python_file')

link_with_capac_list = list(zload('C:/Users/Salomon Wollenstein/Documents/GitHub/InverseVIsTraffic/temp_files/links_with_capac.pkz'))
cnt = 0
jing_link = {}
for link in link_with_capac_list:
    AM_capac = link.AM_capac
    MD_capac = link.MD_capac
    PM_capac = link.PM_capac
    NT_capac = link.NT_capac
    length = link.length
    from_node = link.init_node
    to_node = link.term_node
    jing_link[(from_node, to_node)] =  AM_capac, MD_capac, PM_capac, NT_capac, length
    cnt =+ 1 

#results for Salo's net 
cap_data = pd.read_pickle(out_dir + 'cap_data' + files_ID + '.pkz')


### Flows

os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')
instance = 'NT'
# Convert Salo's flow into Jing 

flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')

list_=[]
b = pd.DataFrame()
for ts in flow_after_conservation.keys():
    a = pd.DataFrame(flow_after_conservation[ts].items(), columns = ['link', 'flow'])
    a['timestamp'] = len(flow_after_conservation[ts]) * [ts]
    b = b.append(a)
    a = []

c = b[b.link == '(3, 5)']
# Read flows form Jing 
import os 
import json
os.chdir('C:/Users/Salomon Wollenstein/Documents/GitHub/InverseVIsTraffic/temp_files/')
link_label_dict_= zload('link_label_dict_.pkz')
with open('../temp_files/link_day_minute_Apr_dict_JSON_adjusted.json', 'r') as json_file:
    link_day_minute_Apr_dict_JSON_ = json.load(json_file)