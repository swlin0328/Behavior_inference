import numpy as np
import pandas as pd

from . import user_load_data
from time import strftime
from main.lib.config.config import DATA_PATH

'''
e.g.
dtype={ 'uuid': str, 'userId': str }
date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
parse_dates=['reportTime']
'''
def load_dataset(file_path, dtype=None, date_parser=None, parse_dates=None, index_col=None):
	file_path = DATA_PATH + file_path
	dataFrame = pd.read_csv(file_path, dtype=dtype, date_parser=date_parser, parse_dates=parse_dates, index_col=index_col)
	return dataFrame

def get_current_time():
	return pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')

def save_csv(dataSet, file_name):
	# file_path = 'data/backup/{}_{}'.format(current_time, file_name)
	# dataSet.to_csv(file_path, encoding='utf-8', index=False)
	# print('save: ' + file_path)
	file_path = DATA_PATH + file_name
	dataSet.to_csv(file_path, encoding='utf-8', index=False)
	print('==== The ' + file_path, ' is saved ====')

def load_preprocess_dataSet():
	dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
	dataSet = load_dataset('for_clustering.csv')
	dataSet['uuid'] = dataSet['uuid'].astype(str)
	dataSet['userId'] = dataSet['userId'].astype(str)
	dataSet = user_load_data.transform_time(dataSet, 'reportTime', format='%Y-%m-%d')
	dataSet.set_index('uuid', inplace=True)
	return dataSet

def get_peroid_column_dataSet(dataSet):
	peroid_column = user_load_data.create_peroid_column()
	dataSet = dataSet[['userId'] + peroid_column][:1000]
	return dataSet
