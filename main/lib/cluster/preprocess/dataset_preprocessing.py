import numpy as np
import pandas as pd

from . import user_load_data
from . import csv_process

def start():
	# 讀取原始 .csv 檔
	dataSet = csv_process.load_dataset('row_dataset.csv')
	dataSet = dataSet.rename(index=str, columns={"userId": "User_ID", "reporttime": "Reporttime", "reportTime": "Reporttime", "channelId":"channelid"})
	dataSet['User_ID'] = dataSet['User_ID'].astype(int).astype(str)
	dataSet['channelid'] = dataSet['channelid'].astype(int)
	dataSet = user_load_data.transform_time(dataSet, 'Reporttime', format='%Y-%m-%d %H:%M:%S')

	# 先做 channelid 0
	dataSet = only_use_channelId_0_dataSet(dataSet)

	# 刪除異常值，因為發現 sensor 本身有問題
	dataSet = delete_outliers_dataSet(dataSet)

	# 以 User_ID 分類，彙整每個使用者用電資料為每 15 分鐘一筆，w 四捨五入至小數 2 位
	dataSet = group_dataSet(dataSet)

	# 彙整與轉置多個使用者的用電資料 (96 期)
	dataSet = consolidation_dataSet(dataSet)

	# 缺值處理
	dataSet = process_na_dataSet(dataSet)

	# 計算 最大需量、最大需量、總用電量
	dataSet = calc_peroid_max_min_sum_w(dataSet)
	csv_process.save_csv(dataSet, 'for_clustering.csv')
	return dataSet

# 先做 channelid 0
def only_use_channelId_0_dataSet(dataSet):
	dataSet = dataSet[dataSet['channelid'] == 0]
	return dataSet

# 刪除異常值，因為發現 sensor 本身有問題
def delete_outliers_dataSet(dataSet):
	dataSet = user_load_data.delete_outliers(dataSet)
	# 改變 'Reporttime' 欄位 type (string to datetime)
	dataSet = user_load_data.transform_time(dataSet, column='Reporttime', format='%Y-%m-%d %H:%M:%S')
	return dataSet

# 以 User_ID 分類，彙整每個使用者用電資料為每 15 分鐘一筆，w 四捨五入至小數 2 位
def group_dataSet(dataSet):
	dataSet = user_load_data.groupbyData(dataSet, column='User_ID')
	return dataSet

# 彙整與轉置多個使用者的用電資料 (96 期)
def consolidation_dataSet(dataSet):
	# 建立彙整資料欄位
	peroid_column = user_load_data.create_consolidation_column()
	consolidation_dataSet_list = user_load_data.consolidation_all_dataSet(dataSet)
	dataSet = pd.DataFrame(consolidation_dataSet_list, columns=peroid_column)
	return dataSet

# 缺值處理
def process_na_dataSet(dataSet):
	# 建立彙整資料欄位
	peroid_column = user_load_data.create_consolidation_column()
	dataSet = user_load_data.process_na(dataSet, peroid_column, threshold=2)
	return dataSet

# 計算 最大需量、最大需量、總用電量
def calc_peroid_max_min_sum_w(dataSet):
	dataSet = user_load_data.peroid_max_min_sum_w(dataSet)
	return dataSet
