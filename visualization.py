from utils import *
import json
import numpy as np
import os 
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import pandas as pd

def histo_flows(instance, out_dir, files_ID, year, month, week_day_list):
    link_min_dict = zload(out_dir + "link_min_dict" + files_ID + ".pkz")
    link_id = zload(out_dir + "link_edge_dict" + files_ID + ".pkz")
    capacity = zload(out_dir + "capacity_link" + files_ID + ".pkz")
    
    for i in link_id.keys():
        flow = []
        for day in week_day_list:
            a = link_min_dict["link_" + str(link_id[i]) + "_" + str(year) + "_" + str(month) + "_" + str(day)]
            a = a["flow_" + instance]
            flow.extend(a)
        capac = capacity[str(link_id[i]) + "_" + instance]
        
        plt.figure()
        plt.title('link' +  str(link_id[i]) + " :" + instance )
        plt.axvline(capac, color='k', linestyle='dashed', linewidth=2)
        plt.hist(flow, color='c', edgecolor='k', alpha=0.65)
        
            
    

def plot_POA(instance, out_dir, month_w):
#for instance in time_instances['id']:
	#with open(out_dir + "PoA_dict_noAdj_" + month_w + '_' + instance + '.json', 'r') as json_file:
#		PoA_dict_noAdj = json.load(json_file)
	#PoA_dict_noAdj = sorted(PoA_dict_noAdj.items())
	#x, y = zip(*PoA_dict_noAdj)

	PoA_dict = {}

	#for i in range(len(x)):
	#	PoA_dict[int(x[i])] = y[i]
	

	with open(out_dir + "PoA_dict_" + month_w + '_' + instance + '.json', 'r') as json_file:
		PoA_dict_ = json.load(json_file)
	PoA_dict_ = sorted(PoA_dict_.items())
	x2, y2 = zip(*PoA_dict_)

	PoA_dict2 = {}

	for i in range(len(x2)):
		PoA_dict2[int(x2[i])] = y2[i]
	
	plt.figure()
	PoA_dictP = plt.plot(PoA_dict.keys(), PoA_dict.values(), "bo-")
	PoA_dict_noAdj = plt.plot(PoA_dict2.keys(), PoA_dict2.values(), "rs-")
	#plt.legend([PoA_dict, PoA_dict_noAdj], ["PoA", "PoA demand adj"], loc=0)
	plt.xlabel('Days of ' + month_w)
	plt.ylabel('PoA')
	#pylab.xlim(1, 30)
	#pylab.ylim(0.9, 2.0)
	grid("on")
	savefig(out_dir + 'PoA'+ '_' + instance + '_' + month_w +'.pdf')



def plot_cong(instance, out_dir, month_w):
#for instance in time_instances['id']:
	'''
	with open(out_dir + "cong_dict_noAdj_" + month_w + '_' + instance + '.json', 'r') as json_file:
		cong_dict_noAdj = json.load(json_file)
	cong_dict_noAdj = sorted(cong_dict_noAdj.items())
	x, y = zip(*cong_dict_noAdj)
	'''
	with open(out_dir + "cong_" + month_w + '_' + instance + '.json', 'r') as json_file:
		cong_dict_ = json.load(json_file)

	cong_dict_ = sorted(cong_dict_.items())
	x2, y2 = zip(*cong_dict_)
	cong_dict = {}

	for i in range(len(cong_dict_)):
		cong_dict[int(x2[i])] = y2[i]


	plt.figure()
#	PoA_dict = plt.plot(x, y, "bo-")
	PoA_dict_noAdj = plt.plot(cong_dict.keys(),cong_dict.values() ,  "rs-")
	#plt.legend([PoA_dict, PoA_dict_noAdj], ["PoA", "PoA demand adj"], loc=0)
	plt.xlabel('Days of ' + month_w)
	plt.ylabel('cong')
	#pylab.xlim(-0.1, 1.6)
	#pylab.ylim(0.9, 2.0)
	grid("on")
	savefig(out_dir + 'cong'+ '_' + instance + '_' + month_w +'.pdf')
	#plt.show()


def plt_cong_vs_poa(instance, out_dir, month_w):
	# Load congestion
	with open(out_dir + "cong_" + month_w + '_' + instance + '.json', 'r') as json_file:
		cong_dict_ = json.load(json_file)

	cong_dict_ = sorted(cong_dict_.items())
	x2, y2 = zip(*cong_dict_)
	cong_dict = {}

	for i in range(len(cong_dict_)):
		cong_dict[int(x2[i])] = y2[i]

	#Load PoA
#	with open(out_dir + "PoA_dict_noAdj_" + month_w + '_' + instance + '.json', 'r') as json_file:
		#PoA_dict_noAdj = json.load(json_file)
	#PoA_dict_noAdj = sorted(PoA_dict_noAdj.items())
	#x, y = zip(*PoA_dict_noAdj)

	PoA_dict = {}

	#for i in range(len(x)):
#		PoA_dict[int(x[i])] = y[i]
	

	with open(out_dir + "PoA_dict_" + month_w + '_' + instance + '.json', 'r') as json_file:
		PoA_dict_ = json.load(json_file)
	PoA_dict_ = sorted(PoA_dict_.items())
	x2, y2 = zip(*PoA_dict_)

	PoA_dict2 = {}

	for i in range(len(x2)):
		PoA_dict2[int(x2[i])] = y2[i]


	#Dict relating cong and Poa
	poa_cong_dict = {}
	for key in PoA_dict2.keys():
		poa_cong_dict[PoA_dict2[key]] = cong_dict[key]

	
	plt.figure()
#	PoA_dict = plt.plot(x, y, "bo-")
	PoA_dict_noAdj = plt.scatter(poa_cong_dict.values(),poa_cong_dict.keys())
	#plt.legend([PoA_dict, PoA_dict_noAdj], ["PoA", "PoA demand adj"], loc=0)
	plt.xlabel('Cong ' + month_w)
	plt.ylabel('PoA')
	#pylab.xlim(-0.1, 1.6)
	#pylab.ylim(0.9, 2.0)
	grid("on")
	savefig(out_dir + 'Cong_vs_PoA_'+ instance + '_' + month_w +'.pdf')
	#plt.show()



def plt_cong_vs_all(time_instances, out_dir, month_w):
	# Load congestion
	poa_cong_dict = {}
	plt.figure()
	for instance in time_instances['id']:
		with open(out_dir + "cong_" + month_w + '_' + instance + '.json', 'r') as json_file:
			cong_dict_ = json.load(json_file)

		cong_dict_ = sorted(cong_dict_.items())
		x2, y2 = zip(*cong_dict_)
		cong_dict = {}

		for i in range(len(cong_dict_)):
			cong_dict[int(x2[i])] = y2[i]

		#Load PoA
	#	with open(out_dir + "PoA_dict_noAdj_" + month_w + '_' + instance + '.json', 'r') as json_file:
#			PoA_dict_noAdj = json.load(json_file)
		#PoA_dict_noAdj = sorted(PoA_dict_noAdj.items())
		#x, y = zip(*PoA_dict_noAdj)

		PoA_dict = {}

		for i in range(len(x2)):
			PoA_dict[int(x2[i])] = y2[i]
		

		with open(out_dir + "PoA_dict_" + month_w + '_' + instance + '.json', 'r') as json_file:
			PoA_dict_ = json.load(json_file)
		PoA_dict_ = sorted(PoA_dict_.items())
		x2, y2 = zip(*PoA_dict_)

		PoA_dict2 = {}

		for i in range(len(x2)):
			PoA_dict2[int(x2[i])] = y2[i]


		#Dict relating cong and Poa
		poa_cong_dict={}
		for key in PoA_dict.keys():
			poa_cong_dict[PoA_dict2[key]] = cong_dict[key]

	
	
		#	PoA_dict = plt.plot(x, y, "bo-")
		PoA_dict_noAdj = plt.scatter(poa_cong_dict.values(),poa_cong_dict.keys(), alpha = 0.7, label= instance)
		plt.legend(loc=0)
		#plt.legend(PoA_dict_noAdj, instance, loc=0)
		plt.xlabel('Cong ' + month_w)
		plt.ylabel('PoA')
		#pylab.xlim(-0.1, 1.6)
		#pylab.ylim(0.9, 2.0)
		grid("on")
		savefig(out_dir + 'Cong_vs_PoA_all'+ '_' + month_w +'.pdf')
		#plt.show()


def plt_obj_vs_all(time_instances, out_dir, month_w):
	# Load congestion
	poa_cong_dict = {}
	plt.figure()
	for instance in time_instances['id']:
		with open(out_dir + "obj_dict" + month_w + '_' + instance + '.json', 'r') as json_file:
			obj_dict = json.load(json_file)
		obj_dict = sorted(obj_dict.items())
		x2, y2 = zip(*obj_dict)
		#print len(x2)
		obj_dict = {}

		for i in range(len(x2)):
			obj_dict[int(x2[i])] = y2[i]

		#print obj_dict
		#Load PoA
		#with open(out_dir + "PoA_dict_noAdj_" + month_w + '_' + instance + '.json', 'r') as json_file:
	#		PoA_dict_noAdj = json.load(json_file)
	#	PoA_dict_noAdj = sorted(PoA_dict_noAdj.items())
#		x, y = zip(*PoA_dict_noAdj)

		PoA_dict = {}

		for i in range(len(x2)):
			PoA_dict[int(x2[i])] = y2[i]
		

		with open(out_dir + "PoA_dict_" + month_w + '_' + instance + '.json', 'r') as json_file:
			PoA_dict_ = json.load(json_file)
		PoA_dict_ = sorted(PoA_dict_.items())
		x2, y2 = zip(*PoA_dict_)

		PoA_dict2 = {}

		for i in range(len(x2)):
			PoA_dict2[int(x2[i])] = y2[i]

		with open(out_dir + "cong_" + month_w + '_' + instance + '.json', 'r') as json_file:
			cong_dict_ = json.load(json_file)

		cong_dict_ = sorted(cong_dict_.items())
		x2, y2 = zip(*cong_dict_)
		cong_dict = {}
		max_cong = max(y2)	
		cong_dict_marker = {}
		for i in range(len(cong_dict_)):
			cong_dict[int(x2[i])] = y2[i]
			cong_dict_marker[int(x2[i])] = y2[i]/max_cong
		
		#Dict relating cong and Poa
		poa_cong_dict={}
		for key in PoA_dict2.keys():
			#print key
			poa_cong_dict[PoA_dict2[key]] = obj_dict[key]

		#print(poa_cong_dict)
		#poa_cong_dict = sorted(poa_cong_dict.items())
		#cong_dict = sorted(cong_dict.items())
		
		#	PoA_dict = plt.plot(x, y, "bo-")
		PoA_dict_noAdj = plt.scatter(poa_cong_dict.values(),poa_cong_dict.keys(), alpha = 0.7, label= instance)
		plt.legend(loc=0)
		#plt.legend(PoA_dict_noAdj, instance, loc=0)
		plt.xlabel('Obj diff ' + month_w)
		plt.ylabel('PoA')
		#pylab.xlim(-0.1, 1.6)
		#pylab.ylim(0.9, 2.0)
		grid("on")
		savefig(out_dir + 'Obj_Diff_vs_PoA_all'+ '_' + month_w +'.pdf')
		#plt.show()

def plt_obj_vs_cong_all(time_instances, out_dir, month_w):
	# Load congestion
	poa_cong_dict = {}
	plt.figure()
	for instance in time_instances['id']:
		with open(out_dir + "obj_dict" + month_w + '_' + instance + '.json', 'r') as json_file:
			obj_dict = json.load(json_file)
		obj_dict = sorted(obj_dict.items())
		x2, y2 = zip(*obj_dict)
		#print len(x2)
		obj_dict = {}

		for i in range(len(x2)):
			obj_dict[int(x2[i])] = y2[i]

		#print obj_dict
		#Load PoA
	#	with open(out_dir + "PoA_dict_noAdj_" + month_w + '_' + instance + '.json', 'r') as json_file:
#			PoA_dict_noAdj = json.load(json_file)
		#PoA_dict_noAdj = sorted(PoA_dict_noAdj.items())
		#x, y = zip(*PoA_dict_noAdj)

		PoA_dict = {}

		for i in range(len(x2)):
			PoA_dict[int(x2[i])] = y2[i]
		

		with open(out_dir + "PoA_dict_" + month_w + '_' + instance + '.json', 'r') as json_file:
			PoA_dict_ = json.load(json_file)
		PoA_dict_ = sorted(PoA_dict_.items())
		x2, y2 = zip(*PoA_dict_)

		PoA_dict2 = {}

		for i in range(len(x2)):
			PoA_dict2[int(x2[i])] = y2[i]

		with open(out_dir + "cong_" + month_w + '_' + instance + '.json', 'r') as json_file:
			cong_dict_ = json.load(json_file)

		cong_dict_ = sorted(cong_dict_.items())
		x2, y2 = zip(*cong_dict_)
		cong_dict = {}
		max_cong = max(y2)	
		cong_dict_marker = {}
		for i in range(len(cong_dict_)):
			cong_dict[int(x2[i])] = y2[i]
			
		#Dict relating cong and Poa
		poa_cong_dict={}
		for key in cong_dict.keys():
			#print key
			poa_cong_dict[cong_dict[key]] = obj_dict[key]

		#print(poa_cong_dict)
		#poa_cong_dict = sorted(poa_cong_dict.items())
		#cong_dict = sorted(cong_dict.items())
		
		#	PoA_dict = plt.plot(x, y, "bo-")
		PoA_dict_noAdj = plt.scatter(poa_cong_dict.values(),poa_cong_dict.keys(), alpha = 0.7, label= instance)
		plt.legend(loc=0)
		#plt.legend(PoA_dict_noAdj, instance, loc=0)
		plt.xlabel('Obj diff ' + month_w)
		plt.ylabel('Cong')
		#pylab.xlim(-0.1, 1.6)
		#pylab.ylim(0.9, 2.0)
		grid("on")
		savefig(out_dir + 'Obj_Diff_vs_Cong_all'+ '_' + month_w +'.pdf')
		#plt.show()



def heatmap_ODdemand(out_dir, files_ID,  month_id, instance, week_day_list):
	cnt = 0
	for day in week_day_list:
		cnt = cnt + 1
		OD_demand_dict = {}
		with open(out_dir + 'OD_demands/OD_demand_matrix_' +  month_id + '_full_weekday_'+ instance + files_ID + '.txt', 'r') as the_file:
			num_lines = sum(1 for line in the_file)

		if cnt ==1:
			x = np.zeros(num_lines)

		np.transpose
		with open(out_dir + 'OD_demands/OD_demand_matrix_'+ month_id + "_" + str(day) + '_weekday_'+ instance + files_ID + '.txt', 'r') as the_file:
			for line in the_file:
				sep = line.split(",")
				origin = sep[0]
				dest = sep[1]
				demand = sep[2]
				OD_demand_dict[str(origin) + '->' + str(dest)] = float(demand)
				OD_demand_dict = collections.OrderedDict(sorted(OD_demand_dict.items()))
			a = np.array(list(OD_demand_dict.values()))
			#print(OD_demand_dict)
			x = np.c_[x,a]
	x = np.delete(x,0,1)
	x = np.asmatrix(x)
	x = pd.DataFrame(x)
	x.columns = week_day_list
	x.index = OD_demand_dict.keys()
	#sns.set()
	sns_plot = sns.heatmap(x,  cmap="YlGnBu", linewidths=.1, xticklabels = True)
	fig = sns_plot.get_figure()
	fig.savefig(out_dir + 'OD_demand'+ '_' + instance + '_' + month_w +'.pdf')
	fig.clf()



def heatmap_ODdemand_adj(out_dir, files_ID,  month_w, instance, week_day_list):
	cnt = 0
	for day in week_day_list:
		cnt = cnt + 1
		OD_demand_dict = {}

		with open(out_dir + "demandsDict/demandsDictFixed" + str(day) + "_" + month_w + "_" + instance + ".json", 'r') as json_file:
			OD_demand_dict = json.load(json_file)

		num_lines = len(OD_demand_dict)
		if cnt ==1:
			x = np.zeros(num_lines)

		np.transpose
		for edege in OD_demand_dict:
			OD_demand_dict = collections.OrderedDict(sorted(OD_demand_dict.items()))
		a = np.array(list(OD_demand_dict.values()))
		#print(OD_demand_dict)
		x = np.c_[x,a]
	x = np.delete(x,0,1)
	x = np.asmatrix(x)
	x = pd.DataFrame(x)
	x.columns = week_day_list
	x.index = OD_demand_dict.keys()
	#sns.set()
	sns_plot = sns.heatmap(x,  cmap="YlGnBu", linewidths=.1, xticklabels = True)
	fig = sns_plot.get_figure()
	fig.savefig(out_dir + 'OD_demandFixed'+ '_' + instance + '_' + month_w +'.pdf')
	fig.clf()



def plot_poa_gls(out_dir, files_ID,  month_w, time_instances, week_day_list):
	
	for instance in time_instances['id']:

		with open (out_dir + 'OD_demands/gls_cost_vec_'+ month_w + '_weekday_'+ instance + files_ID + '.json', 'r' ) as json_file:
			gls_cost_vec = json.load(json_file)

		#Load PoA
		#with open(out_dir + "PoA_dict_noAdj_" + month_w + '_' + instance + '.json', 'r') as json_file:
	#		PoA_dict_noAdj = json.load(json_file)
#		PoA_dict_noAdj = sorted(PoA_dict_noAdj.items())
		#x, y = zip(*PoA_dict_noAdj)

		PoA_dict = {}

		for i in range(len(x)):
			PoA_dict[int(x[i])] = y[i]
		

		with open(out_dir + "PoA_dict_" + month_w + '_' + instance + '.json', 'r') as json_file:
			PoA_dict_ = json.load(json_file)
		PoA_dict_ = sorted(PoA_dict_.items())
		x2, y2 = zip(*PoA_dict_)

		PoA_dict2 = {}

		for i in range(len(x)):
			PoA_dict2[int(x2[i])] = y2[i]

		#print(gls_cost_vec)

		#Dict relating gls and Poa
		poa_gls_dict={}
		for key in PoA_dict.keys():
		#	print(key)
			poa_gls_dict[PoA_dict2[key]] = gls_cost_vec[str(key)]

	
		#PoA_dict_noAdj = []
		#	PoA_dict = plt.plot(x, y, "bo-")
		plt_ = plt.scatter(poa_gls_dict.values(),poa_gls_dict.keys(), alpha = 0.7, label= instance)
		plt.legend(loc=0)
		#plt.legend(PoA_dict_noAdj, instance, loc=0)
		plt.xlabel('GLS cost ' + month_w)
		plt.ylabel('PoA')
		#pylab.xlim(-0.1, 1.6)
		#pylab.ylim(0.9, 2.0)
	plt.grid("on")
	fig_ = plt_.get_figure()
	fig_.savefig(out_dir + 'GLS_vs_PoA_all'+ '_' + month_w +'.pdf')
	fig_.clf()


def polyEval(coeff, pt):
	x = sum([coeff[i] * pt^(i-1) for i in range(len(coeff))])
	
def plot_cost_funct(out_dir, files_ID, link, month_w, key, time_instances):
	for instance in time_instances['id']:
		with open(out_dir + "coeffs_dict_"+ month_w + "_" + instance + ".json", 'r') as json_file:
			coeff = json.load(json_file)

		coeff = coeff[key]


def plot_fcoeff(out_dir, month_w, instance):
	with open(out_dir + "fcoeffs_" + month_w + '_' + instance + '.json', 'r') as json_file:
		fcoeff_dict = json.load(json_file)
	
	x = np.linspace(0,1.3,100)

	for day in fcoeff_dict.keys():
		f = []
		#print(fcoeff_dict[day].values()[0][1])
		#print(range(len(fcoeff_dict[day])))
		for i in x:
			f.append(sum([fcoeff_dict[day].values()[0][a]*i**(a) for a in range(len(fcoeff_dict[day].values()[0]))]))
		#print(f)
		plt_ = plt.scatter(x,f, alpha = 0.7, label= str(day))
		plt.legend(loc=0)
		#plt.legend(loc=0)
		#plt.legend(PoA_dict_noAdj, instance, loc=0)
		#plt.xlabel('GLS cost ' + month_w)
		#plt.ylabel('PoA')
		#pylab.xlim(-0.1, 1.6)
		#pylab.ylim(0.9, 2.0)
	plt.grid("on")
	fig_ = plt_.get_figure()
	fig_.savefig(out_dir + 'fcoeffs'+ '_' + instance + '_' +  month_w +'.pdf')
	fig_.clf()

week_day_list = week_day_list[0:21]
#week_day_list = [week_day_list[1]]
print(week_day_list)

for instance in time_instances['id']:
#for instance in ['AM',  'PM']:    
	plot_POA(instance, out_dir, month_w)
	plot_cong(instance, out_dir, month_w)
	plt_cong_vs_poa(instance, out_dir, month_w)
	heatmap_ODdemand(out_dir, files_ID,  month_w, instance, week_day_list)
	heatmap_ODdemand_adj(out_dir, files_ID,  month_w, instance, week_day_list)
	plot_fcoeff(out_dir, month_w, instance)

plt_obj_vs_cong_all(time_instances, out_dir, month_w)
plt_obj_vs_all(time_instances, out_dir, month_w)
plt_cong_vs_all(time_instances, out_dir, month_w)




''' 

plot_poa_gls(out_dir, files_ID,  month_w, time_instances, week_day_list)

links = zload(out_dir + 'G'+ files_ID + ".pkz")
links = list(links.edges())
key = '(6, 2.5, 1000, 1)'

for instance in time_instances['id']:
	for link in links:
		plot_cost_funct(out_dir, files_ID, link, month_w, key, instance)

'''