import numpy as np
import pandas as pd
from lib.util.convert2json import df2json
from lib.util.convert2json import dict2json
from lib.util.convert2json import combine_json
from lib.util.convert2json import combine_json_contain_list
from .dataset_description import dataset_statistics
from sklearn.model_selection import train_test_split
import ast

DATA_DIR = r'../data/'
RESULT_DIR = r'../data/result/'

def load_data(dataset='row_dataset', num_data=10):
	dataset = dataset_statistics(dataset)
	first_N_data = dataset.list_data(num_data)
	attr = dataset.list_attributes()
	result = {'data_attribute': attr, 'data': ast.literal_eval(first_N_data)}
	return dict2json(result, 'dataset_info')

def drop_NA(df, attr):
	df = df.dropna(axis=0, how='any', subset=attr)
	return df

def fill_NA_with_0(df, attr):
	value = 0
	df[attr] = df[attr].fillna(value, axis=0)
	return df

def fill_NA_with_mean(df, attr):
	value = df[attr].mean(axis=0)
	df[attr] = df[attr].fillna(value, axis=0)
	return df

def fill_NA_with_median(df, attr):
	value = df[attr].median(axis=0)
	df[attr] = df[attr].fillna(value, axis=0)
	return df

def data_cleaning(dataSet, method, attr=None):
	data_process = {
		'drop_NA': drop_NA,
		'fill_0': fill_NA_with_0,
		'fill_mean': fill_NA_with_mean,
		'fill_median': fill_NA_with_median
	}
	df = pd.read_csv(DATA_DIR + dataSet + '.csv', header=0)
	if attr is None:
		attr = df.columns.values
	df = data_process[method](df, attr)
	df.to_csv(DATA_DIR + 'data_cleaning.csv', index=False)
	return dict2json({'result': True}, 'result4preprocess')

def Min_Max(df, attr):
	series = df.loc[:, attr]
	min = series.min()
	max = series.max()
	series_standardized = (series-min) / (max-min)
	df.loc[:, attr] = series_standardized
	return df

def Standardization(df, attr):
	series = df.loc[:, attr]
	avg = series.mean()
	stdv = series.std()
	series_standardized = (series - avg) / stdv
	df.loc[:, attr] = series_standardized
	return df

def data_scaling(dataSet, scaling_method):
	scaling_process = {
		'Min-Max': Min_Max,
		'Standardization': Standardization
	}
	df = pd.read_csv(DATA_DIR + dataSet + '.csv', header=0)
	for attr, method in scaling_method.items():
		df = scaling_process[method](df, attr)

	df.to_csv(DATA_DIR + 'data_scaling.csv', index=False)
	dataset = dataset_statistics('data_scaling')
	attr = ['mean', 'std', '50%']
	return dataset.describe_dataset(filename='data_scaling', attr=attr, transpose=True)

def dataset_split(dataSet, stratify, attribute):
	df = pd.read_csv(DATA_DIR + dataSet + '.csv', header=0)
	num_data = 5
	if stratify:
		training_data, testing_data = train_test_split(df, test_size=0.2, stratify=df.loc[:, attribute])
	else:
		training_data, testing_data = train_test_split(df, test_size=0.2)

	training_data.to_csv(DATA_DIR + 'training_data.csv', index=False)
	testing_data.to_csv(DATA_DIR + 'testing_data.csv', index=False)

	train_dataset = dataset_statistics('training_data')
	train_first_N_data = train_dataset.list_data(num_data)

	test_dataset = dataset_statistics('testing_data')
	test_first_N_data = test_dataset.list_data(num_data)
	return combine_json_contain_list(train_first_N_data, test_first_N_data, 'training_data', 'testing_data')

def one_hot_encoding(dataSet, attribute):
	df = pd.read_csv(DATA_DIR + dataSet + '.csv', header=0, encoding='big5')
	df2 = pd.get_dummies(df[attribute])
	df = df.join(df2)
	df = df.drop([attribute], axis=1)

	df.to_csv(DATA_DIR + 'one_hot_result.csv', index=False, encoding='utf_8_sig')
	return df2json(df, 'one_hot_result')

data_process = {
	'mu': drop_NA,
	'fill_0': fill_NA_with_0,
	'fill_mean': fill_NA_with_mean,
	'fill_median': fill_NA_with_median
}

def feature_engineering(dataSet, attribute_1, attribute_2, operator):
	df = pd.read_csv(DATA_DIR + dataSet + '.csv', header=0, encoding='big5')
	new_label = attribute_1 + operator + attribute_2
	if operator == '*':
		result = df[attribute_1].multiply(df[attribute_2], fill_value=0)
	if operator == '/':
		result = df[attribute_1].divide(df[attribute_2], fill_value=0)

	df[new_label] = result
	df.to_csv(DATA_DIR + 'feature_engineering.csv', index=False, encoding='utf_8_sig')
	return df2json(df, 'feature_engineering')
