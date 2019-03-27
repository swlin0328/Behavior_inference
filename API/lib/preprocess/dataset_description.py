import os
from lib.util.convert2json import df2json
import pandas as pd
from lib.util.convert2json import combine_json_contain_list

DATA_DIR = r'../data/'
RESULT_DIR = r'../data/result/'

class dataset_statistics():
	""" convert the description of raw dataset to json format.
	Args :
		dataSet (str) : The filename of raw dataset which would be described.

	Return :
		The related result saved in the json format.
	"""
	def __init__(self, dataSet='row_dataset'):
		self.dataSet = pd.read_csv(DATA_DIR + dataSet + '.csv', header=0)

	def list_data(self, num_data = 10):
		temp_df = self.dataSet.head(num_data)
		return df2json(temp_df, 'Dataset_first_' + str(num_data) + '_data', orient='records')

	def list_attributes(self):
		count_nonnull = self.dataSet.notnull().sum(axis=0)
		data_types = self.dataSet.dtypes
		temp_df = pd.DataFrame()
		temp_df.insert(0, u'missing_value', self.dataSet.isnull().any())
		temp_df.insert(0, u'data_type', data_types)
		temp_df.insert(0, u'num_of_non_missing_value', count_nonnull)
		temp_df.insert(0, u'attribute', self.dataSet.columns.values)
		temp_df.to_csv(RESULT_DIR + 'data_attrubute.csv', encoding='utf_8_sig', index=False)
		temp_df = pd.read_csv(RESULT_DIR + 'data_attrubute.csv')

		return df2json(temp_df, 'attrbutes', orient='records')

	def describe_dataset(self, filename='Dataset_statistics', analysis=None, attr=None, transpose=False):
		temp_df = self.dataSet.describe()
		if transpose:
			temp_df = temp_df.T
		if analysis is not None:
			temp_df = temp_df.loc[analysis]
		if attr is not None:
			temp_df = temp_df[attr]
		return df2json(temp_df, filename, orient='index')

def list_dataset():
	""" list the filename of xlsx and csv format in the target folder,
	Args :
		DATA_DIR (str): The target folder which contains the dataset and output result.

	Return :
		json_data {key(str), value}: The key/value pairs of json, and the value is the base64_string encoded with the list of dataset name.
	"""
	dataset_names = []
	for filename in os.listdir(DATA_DIR):
		extension = filename.split('.')[-1]
		if extension == 'xlsx' or extension == 'csv':
			dataset_names.append(filename)

	temp_df = pd.DataFrame({'dataset_names': dataset_names})
	return df2json(temp_df, 'Dataset_names', orient='records')