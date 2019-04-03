import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from time import strftime
import os

from ..cluster.preprocess import csv_process
from ..cluster.model_storage import storage4cluster

class user_dailyload_to_group():
	def __init__(self, dataSet, model_name):
		self.dataSet = dataSet.copy()
		self.model_name = model_name

	def start(self):
		dataSet = self.group_user_dailyload()
		return dataSet

	def group_user_dailyload(self):
		df_user_group = self.dataSet.groupby('User_ID')
		userID = df_user_group.groups.keys()
		model_storage = storage4cluster(model_name=self.model_name)
		KMeans_model = model_storage.load_model_from_sql()

		user_dailyload_group = pd.DataFrame()
		for user in userID:
			dailyload2group = pd.DataFrame()
			user_group = df_user_group.get_group(user)
			input_X = user_group.iloc[:, 5:101] # recorded power consumption
			group = KMeans_model.predict(input_X)

			dailyload2group['Week_ID'] = user_group['Week_ID']
			dailyload2group['Group_ID'] = group
			dailyload2group['Reporttime'] = strftime('%Y-%m-%d %H:%M')
			dailyload2group.insert(loc=0, column='User_ID', value=user)
			user_dailyload_group = user_dailyload_group.append(dailyload2group)

		csv_process.save_csv(user_dailyload_group, 'user_group_relation.csv')
		return user_dailyload_group