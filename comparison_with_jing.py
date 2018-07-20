#comparison with jing's flows
import matplotlib.patches as mpatches

os.chdir('G:/My Drive/Github/InverseVIsTraffic/Python_files')
import util
from util import *


os.chdir('G:\My Drive\Github\PoA\Price_of_Anarchy_for_Transportation_Networks')

jing_folders = ['G:/My Drive/Github/InverseVIsTraffic/temp_files/',
'G:/My Drive/Github/InverseVIsTraffic_zero/temp_files/',
'G:/My Drive/Github/InverseVIsTraffic1/temp_files/']

salo_folder = out_dir

number_plots = 2

#### -------------- Percentiles at TMC Level ---------------------------------------------------------------------------
print('Percentiles')

for folder in jing_folders:

    folder_jing = folder
    
    free_flow_speed_salo = pd.read_pickle(out_dir + 'free_flow_speed_ ' + files_ID + '.pkz')
    free_flow_speed_salo = free_flow_speed_salo.to_dict()
    free_flow_speed_jing = zload(folder_jing + 'tmc_ref_speed_dict.pkz')
    
    
    ds = [free_flow_speed_salo, free_flow_speed_jing]
    d = {}
    for k in free_flow_speed_salo.iterkeys():
        d[k] = list(d[k] for d in ds)
    
    df = pd.DataFrame()
    df = df.from_dict(d).T
    df = df.rename(columns={0: "salo", 1: "jing"})
    df.plot()
    plt.title('TMC free flow 95% -' +  folder_jing[-30:])
    
    plt.figure()
    print()
    plt.plot(df.salo.tolist(),df.jing.tolist(),'.' )
    plt.title('scatter percentiles -' + folder_jing[-30:] )


'''
#### -------------- Speed comparison  ---------------------------------------------------------------------------
print('Speed Profiles')

os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')

for i in range(number_plots):
    for folder in jing_folders:
    
        folder_jing = folder
        #jing
        tmc_ref_speed_dict = zload('G:/My Drive/Github/InverseVIsTraffic1/temp_files/tmc_ref_speed_dict.pkz')
        tmc_day_capac_flow_dict = zload( folder_jing + 'tmc_day_capac_flow_dict_Apr.pkz')
        link_with_capac_list = list(zload(folder_jing + 'links_with_capac.pkz'))
        tmc_net_list = zload(out_dir + 'tmc_net_list' + files_ID + '.pkz')
        
        # load tmc-day-ave_speed data for AM peak of April
        tmc_day_speed_dict_Apr_AM = zload(folder_jing + 'Apr_AM/tmc_day_speed_dict.pkz')
        
        tmc_length_dict = zload( folder_jing + 'link_dicts.pkz')
        a = []
        link_with_capac =  link_with_capac_list[1]
        #tmc = '129P04114'
        #day = 18
        tmc = np.random.choice(tmc_net_list)
        day = np.random.randint(1,30)
        #salo
        instance = 'AM'
        df = pd.read_pickle(out_dir + 'filtered_tmc_date_time_flow' + files_ID + '_' + instance +'.pkz')
        df = df[(df['measurement_tstamp'] > '2012-04-'+str(day)) & (df['measurement_tstamp'] < '2012-04-'+str(day+1))]
        df = df[df['tmc_code'] == tmc]
        df = df['speed']
        
        #speed
        jing_speed = tmc_day_speed_dict_Apr_AM[tmc + str(day)].speed
        salo_speed = df.tolist()[:-1]
        
        plt.figure()
        plt.plot(jing_speed, label="jing_speed")
        plt.plot(salo_speed, label="salo_speed")
        plt.title('Speed Conservation '+ instance + ' tmc: '+ tmc + ' day: ' + str(day)  +  folder_jing[-30:] )
        plt.legend()



#### -------------  Flows before conservation  (TMC level) ---------------------------------------------------------------------------
instance = 'AM'
os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')

tmc_net_list = zload(out_dir + 'tmc_net_list' + files_ID + '.pkz')
df1 = pd.read_pickle(out_dir + 'filtered_tmc_date_time_flow' + files_ID + '_' + instance +'.pkz')
os.chdir('G:/My Drive/Github/InverseVIsTraffic/Python_files')
import util
from util import *
os.chdir('G:/My Drive/Github/InverseVIsTraffic1/000_ETA')
import util
from util import *


os.chdir('G:/My Drive/Github/InverseVIsTraffic/Python_files')

#flow 
#tmc_day_capac_flow_minute_dict = zload('G:/My Drive/Github/InverseVIsTraffic1/temp_files/tmc_day_capac_flow_minute_dict_Apr.pkz')
#road_seg_inr_capac = zload('G:/My Drive/Github/InverseVIsTraffic/temp_files/road_seg_inr_capac.pkz')
tmc_day_capac_flow_dict = zload('G:/My Drive/Github/InverseVIsTraffic/temp_files/tmc_day_capac_flow_dict_Apr.pkz')
tmc_ref_speed_dict = zload('G:/My Drive/Github/InverseVIsTraffic/temp_files/tmc_ref_speed_dict.pkz')

#tmc_day_capac_flow_dict_Apr.pkz
tmc = np.random.choice(tmc_net_list)
day = np.random.randint(1,30)

#jing_flow = tmc_day_capac_flow_minute_dict[tmc + str(day)].AM_flow_minute()
jing_flow = tmc_day_capac_flow_dict[tmc + str(day)].AM_flow()


os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')
df = df1[(df1['measurement_tstamp'] > '2012-04-'+str(day)) & (df1['measurement_tstamp'] < '2012-04-'+str(day+1))]
df = df[df['tmc_code'] == tmc]
df = df['xflow']
salo_flow = df.tolist()[:-1]

plt.figure()
plt.plot(jing_flow, label="jing")
plt.plot(salo_flow, label="salo")
plt.title('Flows before Conservation '+ instance + ' tmc:'+ tmc + 'day:' + str(day)   )
plt.legend()



#### -------------- Capacity of TMCs -----------------------------------------------------------------
for folder in jing_folders:
    instance = 'AM'
    os.chdir(folder)
    os.chdir('../Python_files')
    road_seg_inr_capac = zload('../temp_files/road_seg_inr_capac.pkz')
    os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')
    result2 = pd.read_pickle(out_dir + 'result_2' + files_ID + '_' + instance +'.pkz')
    result2 = result2[~result2.index.duplicated(keep='first')]
    cap_jing = []
    cap_salo = []
    for i in range(len(road_seg_inr_capac.tmc)):
        tmc = road_seg_inr_capac.tmc[i]
        road_num = road_seg_inr_capac.road_num[i]
        shape_length = road_seg_inr_capac.shape_length[i]
        day = day
        AB_capac = road_seg_inr_capac.AB_AM_capac[i]
        #AB_capac = road_seg_inr_capac.AB_MD_capac[i]
        #AB_capac = road_seg_inr_capac.AB_PM_capac[i]
        #AB_capac = road_seg_inr_capac.AB_NT_capac[i]
        
        result3 = result2[result2.index == tmc]
        if len(result3)>0:
            AB_AM_salo_cap =  result3['AB_AMCAPAC'].iloc[0]
            cap_jing.append(AB_capac)
            cap_salo.append(AB_AM_salo_cap)
    
    plt.figure()
    plt.plot(cap_jing, label="cap_jing")
    plt.plot(cap_salo, label="cap_salo")
    plt.title('Capacity of TMCs at '+ instance )
    plt.legend()
    
    plt.figure()
    plt.plot(cap_jing,cap_salo,'.' )
    plt.title('scatter capacities' )

'''
#### -------------  Flows before conservation  (Link level) ---------------------------------------------------------------------------

import json

os.chdir(jing_folders[0])

link_label_dict_ = zload('link_label_dict_.pkz')

with open('../temp_files/link_day_minute_Apr_dict_JSON.json', 'r') as json_file:
    link_day_minute_Apr_dict_JSON = json.load(json_file)



for i in range(number_plots):
    a = np.random.choice(link_day_minute_Apr_dict_JSON.keys())
    day = link_day_minute_Apr_dict_JSON[a]['day']
    link_idx = link_day_minute_Apr_dict_JSON[a]['link_idx']
    
    for folder in jing_folders:
        
        os.chdir(folder)
        
        with open('../temp_files/link_day_minute_Apr_dict_JSON.json', 'r') as json_file:
            link_day_minute_Apr_dict_JSON = json.load(json_file)
            
        os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')
        instance = 'AM'
        
        dicti = {}
        dicti[0] = '(1, 2)'
        dicti[2] = '(1, 3)'
        dicti[1] = '(2, 1)'
        dicti[4] = '(2, 3)'
        dicti[6] = '(2, 4)'
        dicti[3] = '(3, 1)'
        dicti[5] = '(3, 2)'
        dicti[8] = '(3, 5)'
        dicti[10] = '(3, 6)'
        dicti[7] = '(4, 2)'
        dicti[12] = '(4, 5)'
        dicti[16] = '(4, 8)'
        dicti[9] = '(5, 3)'
        dicti[13] = '(5, 4)'
        dicti[14] = '(5, 6)'
        dicti[18] = '(5, 7)'
        dicti[11] = '(6, 3)'
        dicti[15] = '(6, 5)'
        dicti[20] = '(6, 7)'
        dicti[19] = '(7, 5)'
        dicti[21] = '(7, 6)'
        dicti[22] = '(7, 8)'
        dicti[17] = '(8, 4)'
        dicti[23] = '(8, 7)'
        
        flow_before_conservation = pd.read_pickle(out_dir + 'flows_before_QP' + files_ID + '_' + instance +'.pkz')
        
        AM_flow_minute = link_day_minute_Apr_dict_JSON[a]['AM_flow_minute']
        MD_flow_minute = link_day_minute_Apr_dict_JSON[a]['MD_flow_minute']
        PM_flow_minute = link_day_minute_Apr_dict_JSON[a]['PM_flow_minute']
        NT_flow_minute =link_day_minute_Apr_dict_JSON[a]['NT_flow_minute']
        
        
        b = flow_before_conservation
        b['link'] = b['link'].astype(str)
        c = b[b.link == dicti[link_idx]]
        c = c[(c['measurement_tstamp'] > '2012-04-'+str(day)) & (c['measurement_tstamp'] < '2012-04-'+str(day+1))]
        c = c.sort_values(by=['measurement_tstamp'])
        c = c['flow'].tolist()
        c2 = AM_flow_minute
        
        plt.figure()
        plt.plot(c2, label="flow_jing")
        plt.plot(c, label="flow_salo")
        plt.title('Flow before adj '+ instance + ' link:' + dicti[link_idx] + ' day:' + str(day ))
        plt.legend()



#### -------------  Flows after conservation (Link level)

        # Read flows form Jing 
        import os 
        import json
        os.chdir(jing_folders[0])
        link_label_dict_= zload('link_label_dict_.pkz')
        with open('../temp_files/link_day_minute_Apr_dict_JSON_adjusted.json', 'r') as json_file:
            link_day_minute_Apr_dict_JSON_ = json.load(json_file)
        
    #    a = np.random.choice(link_day_minute_Apr_dict_JSON_.keys())
    #    day = link_day_minute_Apr_dict_JSON_[a]['day']
    #    link_idx = link_day_minute_Apr_dict_JSON_[a]['link_idx']
        
        os.chdir(folder)
        with open('../temp_files/link_day_minute_Apr_dict_JSON_adjusted.json', 'r') as json_file:
            link_day_minute_Apr_dict_JSON_ = json.load(json_file)
        
        os.chdir('G:/My Drive/Github/PoA/Price_of_Anarchy_for_Transportation_Networks')
        instance = 'AM'
        # Convert Salo's flow into Jing 
        
        dicti = {}
        dicti[0] = '(1, 2)'
        dicti[2] = '(1, 3)'
        dicti[1] = '(2, 1)'
        dicti[4] = '(2, 3)'
        dicti[6] = '(2, 4)'
        dicti[3] = '(3, 1)'
        dicti[5] = '(3, 2)'
        dicti[8] = '(3, 5)'
        dicti[10] = '(3, 6)'
        dicti[7] = '(4, 2)'
        dicti[12] = '(4, 5)'
        dicti[16] = '(4, 8)'
        dicti[9] = '(5, 3)'
        dicti[13] = '(5, 4)'
        dicti[14] = '(5, 6)'
        dicti[18] = '(5, 7)'
        dicti[11] = '(6, 3)'
        dicti[15] = '(6, 5)'
        dicti[20] = '(6, 7)'
        dicti[19] = '(7, 5)'
        dicti[21] = '(7, 6)'
        dicti[22] = '(7, 8)'
        dicti[17] = '(8, 4)'
        dicti[23] = '(8, 7)'
        
    
        flow_after_conservation = pd.read_pickle(out_dir + 'flows_after_QP' + files_ID + '_' + instance +'.pkz')
        
        AM_flow_minute = link_day_minute_Apr_dict_JSON_[a]['AM_flow_minute']
        MD_flow_minute = link_day_minute_Apr_dict_JSON_[a]['MD_flow_minute']
        PM_flow_minute = link_day_minute_Apr_dict_JSON_[a]['PM_flow_minute']
        NT_flow_minute =link_day_minute_Apr_dict_JSON_[a]['NT_flow_minute']
        
     
        list_=[]
        b = pd.DataFrame()
        for ts in flow_after_conservation.keys():
            d = pd.DataFrame(flow_after_conservation[ts].items(), columns = ['link', 'flow'])
            d['timestamp'] = len(flow_after_conservation[ts]) * [ts]
            b = b.append(d)
            d = []
        
        c = b[b.link == dicti[link_idx]]
        c = c[(c['timestamp'] > '2012-04-'+str(day)) & (c['timestamp'] < '2012-04-'+str(day+1))]
        c = c.sort_values(by=['timestamp'])
        c = c['flow'].tolist()
        c2 = AM_flow_minute
        
        plt.figure()
        plt.plot(c2, label="flow_jing")
        plt.plot(c, label="flow_salo")
        plt.title('Flow adjusted '+ instance + ' link:' + dicti[link_idx] + ' day:' + str(day ))
        plt.legend()


'''
#### ----- read raw files rows length -------------------------- ####

files_dir_salo = 'G:/My Drive/Github/PoA/results/_cdc_all_apr_2012_zero/filtered_tmc_data' 
len_files_salo ={}
for root,dirs,files in os.walk(files_dir_salo):
	    for file in files:
	        df = pd.DataFrame()
	        if file.endswith(".pkz"):
	            df = pd.read_pickle(root + '/' +  file)
                len_files_salo[file] = len(df)

files_dir_jing = 'G:/Team Drives/MPO 2012/raw/Apr' 
len_files_jing ={}
for root,dirs,files in os.walk(files_dir_jing):
	    for file in files:
	        df = pd.DataFrame()
	        if file.endswith(".csv"):
	            df = pd.read_csv(root + '/' +  file)
                len_files_jing[file] = len(df)

'''

### ----------- Comparison of OD-Demands --------------


