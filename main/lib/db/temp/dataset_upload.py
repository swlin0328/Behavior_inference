# coding: utf-8

import pandas as pd
import numpy as np
from ..db.sql_connect import sql_config
import datetime


class dataset4DB():
	def __init__(self, dataset, sql_engine=None,
				 user="", password="", database="", host_address='', port=''):
		self.dataset = pd.read_csv(dataset + '.csv', encoding='utf_8_sig')
		self.init_required_columns()
		self.sql_engine = sql_engine
		if self.sql_engine is None:
			self.sql_config = sql_config(user, password, database, host_address, port).engine

	def chk_df_empty(self, df):
		if df.empty:
			return True
		return False

	def init_required_columns(self):
		cols = self.dataset.columns
		self.kwh_consumption = cols.str.startswith('Issue')
		self.appliances = cols.str.startswith('Num')
		self.meter_ID = cols.str.startswith('User')
		self.location = cols.str.startswith('Location')

	def read_kwh_consumption_table(self):
		required_cols = self.meter_ID | self.kwh_consumption
		temp_df = self.dataset.loc[:, required_cols]
		temp_cols = temp_df.columns

		new_df = pd.DataFrame([], columns=['User_ID', 'Kwh_Consumption', 'Created_Time'])
		for col_name in temp_cols[1:]:
			power_df = temp_df.loc[:, ['User_ID', col_name]]
			month = int(col_name.split('_')[1])
			created_time = datetime.datetime(2018, month*2, 1)
			power_df['Created_Time'] = created_time
			power_df.rename({col_name: 'Kwh_Consumption'}, axis='columns', inplace=True)
			new_df = new_df.append(power_df)

		self.upload_dataset2sql(new_df, 'Power_Consumption')

	def read_appliances_table(self):
		required_cols = self.meter_ID | self.appliances
		temp_df = self.dataset.loc[:, required_cols]
		self.upload_dataset2sql(temp_df, 'Appliances')

	def read_AMI_table(self):
		required_cols = self.meter_ID | self.location
		temp_df = self.dataset.loc[:, required_cols]
		self.upload_dataset2sql(temp_df, 'AMI')

	def read_customer_table(self):
		cols = self.dataset.columns
		cols_select = np.array([True] * cols.size)
		required_cols = cols_select & np.logical_not(
			self.appliances) & np.logical_not(self.kwh_consumption) & np.logical_not(self.location)

		temp_df = self.dataset.loc[:, required_cols]
		self.upload_dataset2sql(temp_df, 'Customer_Info')

	def upload_dataset2sql(self, df, table_name):
		df.to_sql(table_name, self.engine, if_exists='replace', index=False)

	def start(self):
		self.read_kwh_consumption_table()
		self.read_appliances_table()
		self.read_AMI_table()
		self.read_customer_table()
