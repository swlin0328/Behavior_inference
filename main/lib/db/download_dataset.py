# coding: utf-8

import pandas as pd
from ..db.sql_connect import sql_config
from ..config.config import DATA_PATH


class DB2dataset():
	def __init__(self, request_config, sql_conn=None,
				 user="", password="", database="", host_address='', port=''):
		self.sql_config = sql_config(user, password, database, host_address, port, sql_conn)
		self.request_config = request_config

	def chk_df_empty(self, df):
		if df.empty:
			return True
		return False

	def read_data(self, sql_query):
		df = pd.read_sql(sql_query, self.sql_config.db)
		self.sql_config.commit()
		return df

	def read_AMI_table(self):
		sql_query = "select * from AMI"
		df = self.read_data(sql_query)
		#df = df.drop(['Created_Time'], axis=1)
		return df

	def read_kwh_consumption_table(self):
		sql_query = "select * from Power_Consumption"
		df = self.read_data(sql_query)
		return df

	def read_customer_table(self):
		sql_query = "select * from Customer_Info"
		df = self.read_data(sql_query)
		#df = df.drop(['Created_Time'], axis=1)
		return df

	def read_appliances_table(self):
		sql_query = "select * from Appliances"
		df = self.read_data(sql_query)
		#df = df.drop(['Created_Time'], axis=1)
		return df

	def merge_table(self, df1, df2, lkey, rkey):
		merge_df = df1.merge(df2, left_on=lkey, right_on=rkey)
		return merge_df

	def merge_all_tables(self):
		result_df = self.read_AMI_table()
		if self.request_config['customer'] == 'True':
			customer_table = self.read_customer_table()
			result_df = self.merge_table(result_df, customer_table, 'User_ID', 'User_ID')

		if self.request_config['appliances'] == 'True':
			appliances_table = self.read_appliances_table()
			result_df = self.merge_table(result_df, appliances_table, 'User_ID', 'User_ID')

		if self.request_config['kwh'] == 'True':
			kwh_consumption_table = self.generate_ordered_kwh_consumption()
			result_df = self.merge_table(result_df, kwh_consumption_table, 'User_ID', 'User_ID')
		return result_df

	def generate_ordered_kwh_consumption(self):
		target_cols = self.generate_kwh_issues()
		result_df = pd.DataFrame([], columns=target_cols)

		kwh_consumption_table = self.read_kwh_consumption_table()
		user_groups = kwh_consumption_table.groupby('User_ID')

		for user in user_groups.groups.keys():
			user_kwh = user_groups.get_group(user)
			kwh = user_kwh['Kwh_Consumption'].tolist()
			data = [user] + kwh
			data_df = pd.DataFrame([data], columns=target_cols)
			result_df = result_df.append(data_df)

		result_df['User_ID'] = result_df['User_ID'].apply(int)
		return result_df

	def generate_kwh_issues(self):
		user = ['User_ID']
		issues = ['Issue_' + str(idx) for idx in range(1, 7)]
		result = user + issues
		return result

	def start(self):
		result_df = self.merge_all_tables()
		result_df.to_csv(DATA_PATH + 'user_info.csv', encoding='utf_8_sig', index=False)
