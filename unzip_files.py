import os 
import gzip
import pandas as pd
from utils import *
import glob
import csv
'''
def extract_2012_INRIX_data(files_dir, tmc_net_dir, out_dir, files_ID, confidence_score_min,c_value_min):
	#tmc_net_list = zload(tmc_net_dir + 'tmc_net_list' + files_ID + '.pkz')
   tmc_net_list = road_seg_inr_capac.tmc
   cnt = 0
   #filtered_files_list = []
   for root,dirs,files in os.walk(files_dir):
       for file in files:
           df = pd.DataFrame()
           if file.endswith(".gz"):                   
               df = pd.read_csv(root + '/' +  file, compression='gzip', header=0, sep=',', quotechar='"')
               df['year'] = (len(df))*[2012]
               df['measurement_tstamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
               df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute'])
               df = df.rename(columns={'tmc': 'tmc_code', 'c_value': 'cvalue'})
               # Here we will add this function in order to process just once the files
               df = df.set_index('measurement_tstamp')
               df2 = filter_tmc(df,tmc_net_list,confidence_score_min,c_value_min)   
               
               #df2 = df
               #cnt = cnt + 1
               #filtered_files_list.append( out_dir + 'filtered_tmc_date_' + file[:-4]  +'.pkz' )
               pd.to_pickle(df2, out_dir + 'filtered_tmc_' + file[:-4]  +'.pkz')
               #df2.to_csv(out_dir + 'filtered_tmc_' + file[:-4]  +'.csv')
               print(file + ': unzipped !')

   print('-----------------------------------------------------')
'''



def extract_2012_INRIX_data(files_dir, tmc_net_dir, out_dir, files_ID, confidence_score_min,c_value_mi,year):
	#tmc_net_list = zload(tmc_net_dir + 'tmc_net_list' + files_ID + '.pkz')
    road_seg_inr_capac = zload('G:/My Drive/Github/InverseVIsTraffic/temp_files/road_seg_inr_capac.pkz')
    tmc_net_list = road_seg_inr_capac.tmc
    cnt = 0 
    for input_file in glob.glob(files_dir+ '/' + '*.csv.gz'):
        cnt +=1
        with gzip.open(input_file, 'rb') as inp, \
                open(out_dir  + 'filtered_tmc_' + input_file[-22:][:-7] + '.csv', 'wb') as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                if row[0] in road_seg_inr_capac.tmc:
                    writer.writerow(row)
        print(cnt)
    
    for root,dirs,files in os.walk(out_dir +'filtered_tmc_data/'):
        for file in files:
           df = pd.DataFrame()
           if file.endswith(".csv"):                   
               df = pd.read_csv(root + '/' +  file)
               col_names = ['tmc_code', 'month', 'day', 'dow', 'hour', 'minute', 'speed', 'avg_speed', 'ref_speed', 'travel_time', 'confidence_score', 'c_value']
               df.columns = col_names
               df['year'] = (len(df))*[year]
               df['measurement_tstamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
               df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute'])
               df = df.rename(columns={'tmc': 'tmc_code', 'c_value': 'cvalue'})
               # Here we will add this function in order to process just once the files
               df = df.set_index('measurement_tstamp')
               #df2 = filter_tmc(df,tmc_net_list,confidence_score_min,c_value_min)   
               
               df2 = df
               #cnt = cnt + 1
               #filtered_files_list.append( out_dir + 'filtered_tmc_date_' + file[:-4]  +'.pkz' )
               pd.to_pickle(df2, out_dir +'filtered_tmc_data/' + file[:-4]  +'.pkz')
               #df2.to_csv(out_dir + 'filtered_tmc_' + file[:-4]  +'.csv')
               print(file + ': unzipped !')
    