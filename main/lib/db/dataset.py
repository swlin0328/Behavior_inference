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


class sql4Dataset():
	def __init__(self, dataset_name, time_config=None, sql_conn=None,
				 user="", password="", database="", host_address='', port='1433'):
		self.sql_config = sql_config(user, password, database, host_address, port, sql_conn)
		self.dataset_name = dataset_name
		self.time_config = time_config
		self.dataset_id = None
		if self.time_config is None:
			self.time_config = {'start_time': None, 'end_time': None}

	def chk_dataset_exist(self):
		self.search_dataset()
		if self.dataset_id != None:
			print('Warning: dataset_name exist')
			return True
		return False

	def search_dataset(self):
		search_id = "SELECT Dataset_ID FROM Inference_Dataset WHERE Experiment_Name = ?"
		self.sql_config.cursor.execute(search_id, (self.dataset_name, ))
		dataset_id = self.sql_config.cursor.fetchone()
		if dataset_id is not None:
			self.dataset_id = int(dataset_id[0])

	def save2sql(self, pca=None, one_hot=None):
		if not self.chk_dataset_exist():
			self.insert_dataset(pca, one_hot)
			self.sql_config.commit()
		#self.sql_config.disconnect()

	def insert_dataset(self, pca, one_hot):
		add_dataset = "INSERT INTO Inference_Dataset (Experiment_Name, Start_Time, End_Time, Enc_Blob, Created_Time)" \
					" VALUES (?, ?, ?, ?, ?);"
		enc = {'PCA': pca, 'One_Hot': one_hot}
		enc_blob = cPickle.dumps(enc)
		self.sql_config.cursor.execute(add_dataset, (
			self.dataset_name, self.time_config['start_time'], self.time_config['end_time'],
			enc_blob, self.sql_config.created_time))

	def read_dataset_info(self):
		self.sql_config.cursor.execute("SELECT * FROM Inference_Dataset")
		results = self.sql_config.cursor.fetchall()
		for record in results:
			print(record)

	def read_enc_blob(self):
		print('Fetch the target enc blob...')
		sql_cmd = "SELECT Enc_Blob FROM Inference_Dataset WHERE Dataset_ID = ?"
		self.sql_config.cursor.execute(sql_cmd, (self.dataset_id,))
		enc_blob = self.sql_config.cursor.fetchone()
		return enc_blob

	def load_pca_from_sql(self):
		self.search_dataset()
		sql_enc = self.read_enc_blob()
		enc = cPickle.loads(sql_enc[0])
		pca = enc['PCA']
		return pca

	def load_one_hot_from_sql(self):
		self.search_dataset()
		sql_enc = self.read_enc_blob()
		enc = cPickle.loads(sql_enc[0])
		one_hot = enc['One_Hot']
		return one_hot
