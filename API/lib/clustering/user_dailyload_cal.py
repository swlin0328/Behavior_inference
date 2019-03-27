import numpy as np
import pandas as pd
from time import strftime

from lib.util import csv_process

class user_dailyload_cal():
	def __init__(self, dataSet):
		self.dataSet = dataSet.copy()

	def start(self):
		dataSet = self.cal_user_daily_load()
		return dataSet

	def cal_user_daily_load(self):
		self.dataSet['Week_ID'] = pd.to_datetime(self.dataSet['Reporttime']).dt.weekday + 1
		df_user_group = self.dataSet.groupby('User_ID')
		userID = df_user_group.groups.keys()
		users_dailyload = pd.DataFrame()

		for user in userID:
			df_weekday_group = df_user_group.get_group(user).groupby('Week_ID')
			weekday = df_weekday_group.groups.keys()
			dailyload = pd.DataFrame()
			for idx, day in enumerate(weekday):
				target_group_mean = df_weekday_group.get_group(day).iloc[:, 3:99].mean().round(2) # recorded power consumption
				target_group_max = df_weekday_group.get_group(day).loc[:, 'Max_load'].mean().round(2)
				target_group_min = df_weekday_group.get_group(day).loc[:, 'Min_load'].mean().round(2)
				target_group_sum = df_weekday_group.get_group(day).loc[:, 'Total_load'].mean().round(2)

				col_name = target_group_mean.index
				temp_df = pd.DataFrame(data=target_group_mean.values.reshape(1, col_name.shape[0]), columns=col_name)
				temp_df.insert(loc=0, column='avg_Min_load', value=target_group_min)
				temp_df.insert(loc=0, column='avg_Max_load', value=target_group_max)
				temp_df.insert(loc=0, column='avg_Total_load', value=target_group_sum)
				temp_df.insert(loc=0, column='Week_ID', value=day)
				temp_df.insert(loc=0, column='User_ID', value=user)
				dailyload = dailyload.append(temp_df)

			dailyload['Reporttime'] = strftime('%Y-%m-%d %H:%M')
			users_dailyload = users_dailyload.append(dailyload)

		csv_process.save_csv(users_dailyload, 'user_dailyload.csv')
		return users_dailyload