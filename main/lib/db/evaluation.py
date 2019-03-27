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


class sql4Evaluation():
	def __init__(self, experiment_name, customer_group, sql_conn=None,
				 user="", password="", database="", host_address='', port=''):
		self.sql_config = sql_config(user, password, database, host_address, port, sql_conn)
		self.experiment_name = experiment_name
		self.customer_group = customer_group

		self.search_dataset()
		self.search_model()

	def search_dataset(self):
		search_id = "SELECT Dataset_ID FROM Inference_Dataset WHERE Experiment_Name = ?"
		self.sql_config.cursor.execute(search_id, (self.experiment_name, ))
		dataset_id = self.sql_config.cursor.fetchone()
		if dataset_id is not None:
			self.dataset_id = int(dataset_id[0])

	def search_model(self):
		search_id = "SELECT Model_ID FROM Inference_Model WHERE Model_Name = ? AND Customer_Group = ?"
		self.sql_config.cursor.execute(search_id, (self.experiment_name, self.customer_group, ))
		model_id = self.sql_config.cursor.fetchone()
		if model_id is not None:
			self.model_id = int(model_id[0])

	def save2sql(self, f1_score, acc):
		self.insert_evaluation(f1_score, acc)
		self.sql_config.commit()
		#self.sql_config.disconnect()

	def insert_evaluation(self, f1_score, acc):
		add_dataset = "INSERT INTO Inference_Evaluation (Model_ID, Experiment_Name, Customer_Group, F1_score, " \
					  "Accuracy, Dataset_ID, Created_Time) VALUES (?, ?, ?, ?, ?, ?, ?);"
		self.sql_config.cursor.execute(add_dataset, (
			self.model_id, self.experiment_name, self.customer_group, f1_score, acc, self.dataset_id,
			self.sql_config.created_time))

	def read_evaluation_info(self):
		self.sql_config.cursor.execute("SELECT * FROM Inference_Evaluation")
		results = self.sql_config.cursor.fetchall()
		for record in results:
			print(record)
