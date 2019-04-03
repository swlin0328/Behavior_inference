# coding: utf-8

import _pickle as cPickle
from time import strftime
import pymssql
import keras
import types
import tempfile
import keras.models
import os
from ..db.sql_connect import sql_config


class sql4SVM():
	def __init__(self, model_name, customer_group, creator='Guest', sql_conn=None, user="", password="",
				 database="", host_address='', port='1433', description=''):
		self.sql_config = sql_config(user, password, database, host_address, port, sql_conn)
		self.model_name = model_name
		self.customer_group = customer_group
		self.model_creator = creator
		self.description = description
		self.model_id = None

	def chk_model_exist(self):
		self.search_model()
		if self.model_id != None:
			print('Warning: model_name exist')
			return True
		return False

	def search_model(self):
		search_id = "SELECT Model_ID FROM Inference_Model WHERE Model_Name = ?"
		self.sql_config.cursor.execute(search_id, (self.model_name, ))
		model_id = self.sql_config.cursor.fetchone()
		self.sql_config.commit()
		if model_id is not None:
			self.model_id = int(model_id[0])

	def search_dataset(self):
		search_id = "SELECT Dataset_ID FROM Inference_Dataset WHERE Experiment_Name = ?"
		self.sql_config.cursor.execute(search_id, (self.model_name, ))
		dataset_id = self.sql_config.cursor.fetchone()
		self.sql_config.commit()
		if dataset_id is not None:
			self.dataset_id = int(dataset_id[0])

	def save2sql(self, model, model_type, valid_metrics):
		if not self.chk_model_exist():
			self.search_dataset()
			self.insert_model(model_type, valid_metrics)
			self.search_model()
			self.insert_model_blob(model)
			self.sql_config.commit()
		#self.sql_config.disconnect()

	def insert_model(self, model_type, valid_metrics):
		add_model = "INSERT INTO Inference_Model (Model_Name, Model_Type, Customer_Group, Model_Description, Layers, " \
					"Accuracy, Params, Step, Dataset, Created_Time, Created_User, Created_Group)" \
					" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
		self.sql_config.cursor.execute(add_model, (
			self.model_name, model_type, self.customer_group, self.description, 0, valid_metrics,
			0, 0, self.dataset_id, self.sql_config.created_time, self.model_creator, 0))

	def insert_model_blob(self, trained_model):
		model_blob = cPickle.dumps(trained_model)
		add_blob = "INSERT INTO Inference_Blob (Model_ID, Binary_Data) VALUES (?, ?)"
		self.sql_config.cursor.execute(add_blob, (self.model_id, model_blob))

	def read_model_info(self):
		self.sql_config.cursor.execute("SELECT * FROM Inference_Model")
		results = self.sql_config.cursor.fetchall()
		self.sql_config.commit()
		for record in results:
			print(record)

	def read_model_blob(self):
		print('Fetch the target model...')
		sql_cmd = "SELECT Binary_Data FROM Inference_Blob WHERE Model_ID = ?"
		self.sql_config.cursor.execute(sql_cmd, (self.model_id,))
		model_blob = self.sql_config.cursor.fetchone()
		self.sql_config.commit()
		return model_blob

	def load_model_from_sql(self):
		self.search_model()
		sql_model = self.read_model_blob()
		model = cPickle.loads(sql_model[0])
		return model
