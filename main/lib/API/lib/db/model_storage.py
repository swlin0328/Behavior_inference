# coding: utf-8

import _pickle as cPickle
from time import strftime
import pymssql

class storage4cluster():
	def __init__(self, model_name='',  user="", password="", database="dsm", host_address='111.22.33.44', port='2222'):
		self.model_name = model_name
		self.user = user
		self.password = password
		self.db = database
		self.host = host_address
		self.created_time = strftime('%Y-%m-%d %H:%M:%S')
		self.port = port

	def msSQL_connect(self):
		print('=====================================================')
		print('======== Connect to the remote msSQL server ========')
		print('=====================================================')
		print('Time : {}\n'.format(strftime('%Y-%m-%d_%H_%M')))
		self.db = pymssql.connect(server=self.host, port=self.port, user=self.user, password=self.password, database=self.db, charset="utf8")
		self.cursor = self.db.cursor()

	def init_table(self):
		create_table = """Create table cluster_model
							(ID int NOT NULL IDENTITY(1,1), 
							model_name VARCHAR(40), 
							reporttime datetime NOT NULL,
							model VARBINARY(MAX) not null,   
							PRIMARY KEY (model_name, reporttime));"""
		self.cursor.execute(create_table)
		self.db.commit()

	def insert_model_blob(self, trained_model):
		model_blob = cPickle.dumps(trained_model)
		add_blob = "INSERT INTO cluster_model (model_name, reporttime, model) VALUES (%s, %s, %s)"
		self.cursor.execute(add_blob, (self.model_name, self.created_time, model_blob))
		self.db.commit()

	def read_model_blob(self):
		print('Fetch the target model...')
		sql_cmd = "SELECT model FROM cluster_model WHERE model_name = %s"
		self.cursor.execute(sql_cmd, (self.model_name,))
		model_blob = self.cursor.fetchone()
		return model_blob

	def load_model_from_sql(self):
		self.msSQL_connect()
		sql_model = self.read_model_blob()
		model = cPickle.loads(sql_model[0])
		return model

	def disconnect(self):
		self.db.close()
		print('=====================================================')
		print('============ Close the remote connection ============')
		print('=====================================================')

	def save2sql(self, model):
		self.msSQL_connect()
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