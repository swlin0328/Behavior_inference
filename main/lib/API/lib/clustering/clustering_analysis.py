import pandas as pd
import matplotlib.pyplot as plt
from lib.util.convert2json import to_json
from lib.util.convert2json import combine_json
from lib.util.convert2json import df2json
from lib.clustering.clustering import clustering
from lib.clustering.user_dailyload_cal import user_dailyload_cal
from lib.clustering.user_dailyload_to_group import user_dailyload_to_group
import math

DATA_DIR = r'../data/'
RESULT_DIR = r'../data/result/'

def plot_group_center(size, filename='group_center', cols=3):
	""" Plot the clustering result for group center, and saved with the svg format
	Args :
		filename (str) : The filename of the clustering result
		cols (int) : Number of columns of the subplot figure.

	Return :
		json_data {key(str), value}: The key/value pairs of json, and the value is the base64_string encoded with the subplot file.
	"""
	file_name = DATA_DIR + filename + '.csv'
	center_df = pd.read_csv(file_name, header=0)
	rows = math.ceil(center_df.shape[0] / cols)

	f, axarr = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(size, size))
	for idx in range(center_df.shape[0]):
		row = math.floor(idx / cols)
		col = idx % cols
		group_center = center_df.iloc[idx, 1:-1]
		if rows > 1:
			axarr[row][col].scatter(range(group_center.size), group_center, s=5, c='red', marker='o')
			axarr[row][col].set_title('Group-' + str(idx), fontsize=18)
			axarr[row][col].tick_params(labelsize=12)
		else:
			axarr[col].scatter(range(group_center.size), group_center, s=5, c='red', marker='o')
			axarr[col].set_title('Group-' + str(idx), fontsize=18)
			axarr[col].tick_params(labelsize=12)

	f.text(0.5, 0.03, 'Timestamp', ha='center', fontsize=16)
	f.text(0, 0.5, 'Power (w)', va='center', rotation='vertical', fontsize=16)
	plt.savefig(RESULT_DIR + 'group_center.svg', format="svg")

	target_file = {'group_center': RESULT_DIR + 'group_center.svg'}
	to_json(target_file, 'group_center')

def count_group_member(filename='user_group_relation'):
	file_name = DATA_DIR + filename + '.csv'
	df = pd.read_csv(file_name, header=0)

	group = df.groupby('Group_ID')
	group_keys = group.groups.keys()
	result = pd.DataFrame()
	for group_idx in group_keys:
		temp_group = group.get_group(group_idx).describe()
		result.loc['Count', 'Group_'+str(group_idx)] = temp_group.iloc[0, 0]

	df2json(result, 'group_members', orient='index')

def cluster_result(size):
	plot_group_center(size)
	count_group_member()
	return combine_json(r'group_center', r'group_members')

def cluster_training(dataSet, n_clusters, size):
	preprocess_dataSet = pd.read_csv(DATA_DIR + dataSet + '.csv', header=0)
	cluster_centers_df, cluster_labels_df = clustering(preprocess_dataSet, n_clusters).start(to_sql=False)
	user_dailyload_df = user_dailyload_cal(preprocess_dataSet).start()
	user_dailyload_group = user_dailyload_to_group(user_dailyload_df).start()
	return cluster_result(size)

def group_statistics(filename='group_center'):
	file_name = DATA_DIR + filename + '.csv'
	center_df = pd.read_csv(file_name, header=0)
	center_df = center_df.set_index('Group_ID')
	center_df = center_df.iloc[:, :-1].T
	center_df = center_df.describe()
	new_col_name = []
	for col_name in center_df.columns.values:
		new_col_name.append('Group'+ str(col_name))
	center_df.columns = new_col_name
	center_df = center_df.drop(['count'])
	df2json(center_df, 'group_statistics', orient='index')

def group_analysis(size, model_result):
	plot_group_center(size, model_result)
	group_statistics(model_result)
	return combine_json(r'group_center', r'group_statistics', append_child=r'group_statistics')