import os 
import gzip
import pandas as pd
from utils import *


def extract_2012_INRIX_data(files_dir, tmc_net_dir, out_dir, files_ID, confidence_score_min,c_value_min):

	tmc_net_list = zload(tmc_net_dir + 'tmc_net_list' + files_ID + '.pkz')
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
	            cnt = cnt + 1
	            #filtered_files_list.append( out_dir + 'filtered_tmc_date_' + file[:-4]  +'.pkz' )
	            pd.to_pickle(df2, out_dir + 'filtered_tmc_' + file[:-4]  +'.pkz')
	            print(file + ': unzipped !')

	print('-----------------------------------------------------')
	            