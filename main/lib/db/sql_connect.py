# coding: utf-8

from time import strftime
import pymssql
import os
import sqlalchemy as sqlc


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
			url = 'mssql+pyodbc://{}:{}@{}/{}?driver={}'.format(self.user, self.password, self.host,
																self.db_name, 'SQL Server Native Client 10.0')
			self.engine = sqlc.create_engine(url)
			self.db = self.engine.raw_connection()
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
