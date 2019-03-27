# coding: utf-8

import _pickle as cPickle
from time import strftime
import pymssql
import keras
import types
import tempfile
import keras.models
import os


class sql_config():
	def __init__(self, user, password, database, host_address, port, db_conn=None):
		self.user = user
		self.password = password
		self.db_name = database
		self.host = host_address
		self.created_time = strftime('%Y-%m-%d %H:%M:%S')
		self.port = port
		self.db = db_conn

		self.msSQL_connect()

	def msSQL_connect(self):
		print('=====================================================')
		print('======== Connect to the remote msSQL server ========')
		print('=====================================================')
		print('Time : {}\n'.format(strftime('%Y-%m-%d_%H_%M')))
		if self.db is None:
			self.db = pymssql.connect(server=self.sql_config.host, port=self.sql_config.port, user=self.sql_config.user,
									  password=self.sql_config.password, database=self.sql_config.db, charset="utf8")
		self.cursor = self.db.cursor()

	def commit(self):
		try:
			trans = self.db
			trans.commit()
		except:
			trans.rollback()

	def disconnect(self):
		self.db.close()
		print('=====================================================')
		print('============ Close the remote connection ============')
		print('=====================================================')
