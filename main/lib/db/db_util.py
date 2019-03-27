# coding: utf-8

from time import strftime
import os
from ..db.sql_connect import sql_config
import pandas as pd


class sql4DB():
	def __init__(self, query_config, sql_conn=None,
				 user="", password="", database="", host_address='', port=''):
		self.sql_config = sql_config(user, password, database, host_address, port, sql_conn)
		self.query_config = query_config

	def chk_df_empty(self, df):
		if df.empty:
			return True
		return False

	def read_data(self, sql_query):
		df = pd.read_sql(sql_query, self.sql_config.db)
		return df

	def read_model_info(self):
		search_model = "SELECT * FROM Inference_Model "
		if 'model_id' in self.query_config.keys():
			search_model = search_model + 'WHERE Model_ID = ' + self.query_config['model_id']

		model_df = self.read_data(search_model)
		if not self.chk_df_empty(model_df):
			model_df['Created_Time'] = model_df['Created_Time'].dt.strftime('%Y-%m-%d %H:%M')
		return model_df

	def read_model_type(self, model_id):
		chk_dnn = "SELECT Model_Type FROM Inference_Model WHERE Model_ID = ?"
		self.sql_config.cursor.execute(chk_dnn, (model_id, ))
		model_type = self.sql_config.cursor.fetchone()
		return model_type[0]

	def search_model(self):
		search_id = "SELECT Model_ID FROM Inference_Model WHERE Model_Name = ?"
		self.sql_config.cursor.execute(search_id, (self.query_config['model_name'], ))
		model_id = self.sql_config.cursor.fetchone()
		if model_id is not None:
			return int(model_id[0])

	def read_layer_info(self):
		model_id = self.query_config['model_id']
		if self.read_model_type(model_id) == 'SVM':
			result = {'response': 'Invalid model type!'}
			return result
		else:
			search_layer = "SELECT * FROM Inference_Model_Layer WHERE Model_ID = " + str(model_id)
			layer_df = self.read_data(search_layer)
			return layer_df

	def read_dataset_info(self):
		search_dataset = "SELECT Dataset_ID, Experiment_Name, Start_Time, End_Time, Created_Time FROM Inference_Dataset "
		if 'dataset_id' in self.query_config.keys():
			search_dataset = search_dataset + 'WHERE Dataset_ID = ' + self.query_config['dataset_id']

		dataset_df = self.read_data(search_dataset)
		if not self.chk_df_empty(dataset_df):
			dataset_df['Created_Time'] = dataset_df['Created_Time'].dt.strftime('%Y-%m-%d %H:%M')
			dataset_df['Start_Time'] = dataset_df['Start_Time'].dt.strftime('%Y-%m-%d %H:%M')
			dataset_df['End_Time'] = dataset_df['End_Time'].dt.strftime('%Y-%m-%d %H:%M')
		return dataset_df

	def read_evaluation_info(self):
		search_evaluation = "SELECT * FROM Inference_Evaluation "
		if 'model_id' in self.query_config.keys():
			search_evaluation = search_evaluation + 'WHERE Model_ID = ' + self.query_config['model_id']

		evaluation_df = self.read_data(search_evaluation)
		if not self.chk_df_empty(evaluation_df):
			evaluation_df['Created_Time'] = evaluation_df['Created_Time'].dt.strftime('%Y-%m-%d %H:%M')
		return evaluation_df
