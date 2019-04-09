# coding: utf-8

import _pickle as cPickle
from time import strftime
import pymssql
from ..db.sql_connect import sql_config

class storage4cluster():
	def __init__(self, model_name='',  user="", password="", database="",
				 host_address='', port='1433', sql_conn=None):
		self.sql_config = sql_config(user, password, database, host_address, port, sql_conn)
		self.model_name = model_name

	def init_table(self):
		create_table = """Create table cluster_model
							(ID int NOT NULL IDENTITY(1,1), 
							model_name VARCHAR(40), 
							reporttime datetime NOT NULL,
							model VARBINARY(MAX) not null,   
							PRIMARY KEY (model_name, reporttime));"""
		self.sql_config.cursor.execute(create_table)
		self.sql_config.commit()

	def insert_model_blob(self, trained_model):
		model_blob = cPickle.dumps(trained_model)
		add_blob = "INSERT INTO cluster_model (model_name, reporttime, model) VALUES (?, ?, ?)"
		self.sql_config.cursor.execute(add_blob, (self.model_name, self.created_time, model_blob))
		self.sql_config.commit()

	def read_model_blob(self):
		print('Fetch the target model...')
		sql_cmd = "SELECT model FROM cluster_model WHERE model_name = ? ORDER BY reporttime DESC"
		self.sql_config.cursor.execute(sql_cmd, (self.model_name,))
		model_blob = self.sql_config.cursor.fetchone()
		return model_blob

	def load_model_from_sql(self):
		self.sql_config.msSQL_connect()
		sql_model = self.read_model_blob()
		model = cPickle.loads(sql_model[0])
		return model

	def disconnect(self):
		self.sql_config.close()
		print('=====================================================')
		print('============ Close the remote connection ============')
		print('=====================================================')

	def save2sql(self, model):
		self.sql_config.msSQL_connect()
		self.insert_model_blob(model)

	def load_model_from_pkl(self):
		model_path = 'model/' + self.model_name
		model=None
		with open(model_path, 'rb') as file:
			model = cPickle.load(file)
		return model

	def save_model_to_pkl(self, trained_model):
		file_path = r'model/' + self.model_name
		backup_path = r'model/backup/' + self.model_name + '_' + strftime('%Y-%m-%d_%H-%M')
		with open(file_path, 'wb') as file:
			cPickle.dump(trained_model, file)
		with open(backup_path, 'wb') as file:
			cPickle.dump(trained_model, file)
