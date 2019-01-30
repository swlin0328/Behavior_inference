from time import strftime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from lib.util.user_load_data import create_peroid_column
from lib.util import csv_process
from lib.db.model_storage import storage4cluster

class clustering():
	def __init__(self, dataSet, n_clusters):
		self.dataSet = dataSet.copy()
		self.n_clusters = n_clusters
		self.peroid_column = create_peroid_column()

	def start(self, n_init=25, max_iter=1000, model_name='km_model', to_pkl=True, to_sql=False):
		group_center_df, group_label_df = self.KMeans_clustering(n_init=n_init, max_iter=max_iter, model_name=model_name, to_pkl=to_pkl, to_sql=to_sql)
		return group_center_df, group_label_df

	def KMeans_clustering(self, n_init=25, max_iter=1000, model_name='km_model', to_pkl=True, to_sql=False):
		tmp_dataSet = self.dataSet[['UUID', 'User_ID'] + self.peroid_column]
		tmp_dataSet.set_index('UUID', inplace=True)
		KMeans_fit = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=n_init, max_iter=max_iter)
		KMeans_fit.fit(tmp_dataSet[self.peroid_column])

		self.save_model(KMeans_fit, model_name, to_pkl, to_sql)
		# 將 K-Means 每個集群中心的坐標資料存成 CSV
		group_center_df = self.save_cluster_centers_to_csv(KMeans_fit.cluster_centers_)
		# 將每筆資料的 K-Means 分群標籤存成 CSV
		group_label_df = self.save_cluster_with_label_to_csv(KMeans_fit.labels_)

		return group_center_df, group_label_df

	# 將 K-Means 每個集群中心的坐標資料存成 CSV
	def save_cluster_centers_to_csv(self, cluster_centers):
		# 將資料存成 DataFrame，並將數值做四捨五入
		group_center_df = pd.DataFrame(cluster_centers, columns=self.peroid_column).round(2)
		group_center_df['Group_ID'] = list(range(len(group_center_df)))
		group_center_df['Reporttime'] = strftime('%Y-%m-%d %H:%M:%S')

		# 重新排列欄位順序，讓 Group_ID 欄位變成最前面
		columns = ['Group_ID'] + self.peroid_column + ['Reporttime']
		group_center_df = group_center_df.reindex(columns, axis=1)

		csv_process.save_csv(group_center_df, file_name='group_center.csv')
		return group_center_df

	# 將每筆資料的 K-Means 分群標籤存成 CSV
	def save_cluster_with_label_to_csv(self, cluster_labels):
		self.dataSet['Group_ID'] = cluster_labels
		self.dataSet['Updatetime'] = strftime('%Y-%m-%d %H:%M:%S')

		csv_process.save_csv(self.dataSet, file_name='for_cluster_with_label.csv')
		return self.dataSet

	def save_model(self, KMeans_fit, model_name='km_model', to_pkl=True, to_sql=False):
		# 測試用
		#self.save_model_to_pkl(KMeans_fit, model_name)
		model_storage = storage4cluster(model_name)
		if to_sql:
			model_storage.save2sql(KMeans_fit)
		if to_pkl:
			model_storage.save_model_to_pkl(KMeans_fit)

	# =========== 測試用 ===========
	def save_model_to_pkl(self, model, model_name):
		file_path = 'model/' + model_name
		current_time = strftime('%Y-%m-%d-%H-%M-%S')
		backup_path = 'model/backup/{}_{}'.format(current_time, model_name)
		joblib.dump(model, file_path)
		print('save: ' + file_path)
		joblib.dump(model, backup_path)
		print('save: ' + backup_path)