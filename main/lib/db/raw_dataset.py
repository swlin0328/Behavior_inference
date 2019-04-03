#!/usr/bin/python
# -*- coding: utf-8 -*-
from time import strftime
import pandas as pd
from ..db.sql_connect import sql_config
from ..config.config import DATA_PATH
import os


class sql4data():
	def __init__(self, sql_conn=None, user="", password="", database="",
				 host_address='', port='1433'):
		self.sql_config = sql_config(user, password, database, host_address, port, sql_conn)

	def read_data(self, start_date='', end_date='', file_name='raw_dataset'):
		sql_query = "SELECT * FROM raw_training_data WHERE channelid = 0"
		if start_date == '':
			sql_query = sql_query + " and (reporttime BETWEEN '" + start_date + "' AND '" + end_date + "')"

		print('==== Start to query from the raw_data ====')
		df = pd.read_sql(sql_query, self.sql_config.db)
		df.to_csv(DATA_PATH + file_name + '.csv', index=False)
		print('==== The raw_data.csv is saved ====')

	def start(self, start_date='', end_date='', file_name='row_dataset'):
		self.read_data(start_date, end_date, file_name)
