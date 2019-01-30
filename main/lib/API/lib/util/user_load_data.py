from __future__ import division
import numpy as np
import pandas as pd

# 遇到 負數 直接砍，因為發現 sensor 本身有問題
# 預設刪除 > 10000 或 < 0 的值
def delete_outliers(dataSet, threshold=10000):
	return dataSet[(dataSet['w'] >= 0) & (dataSet['w'] <= threshold)]

# string to datetime
# e.g. format='%Y-%M-%d', or '%Y-%m-%d %H:%M:%S' ...
def transform_time(dataSet, column, format):
	dataSet[column] = pd.to_datetime(dataSet[column], format=format)
	return dataSet

# 以某欄位 (e.g. User_ID) 作為分類
# 彙整每個使用者用電資料為每 15 分鐘一筆，w 四捨五入至小數 2 位
def groupbyData(dataSet, column):
	group_dataSet = dataSet.groupby([column, pd.Grouper(key='Reporttime', freq='15T')])['w'].mean().round(2).reset_index()
	return group_dataSet

# 建立以日為單位之欄位 (96 期)
# return ['1-th', '2-th', ... , '95-th', '96-th']
def create_peroid_column():
	return [str(idx) + '-th' for idx in range(1, 97)]

# 建立彙整資料欄位
# return ['UUID', 'User_ID', 'Reporttime', '1-th', ... , '96-th']
def create_consolidation_column():
	return ['UUID', 'User_ID', 'Reporttime'] + create_peroid_column()

# 建立新的 period 時間 (每 15 分鐘一筆，一天共有 96 筆) list
def create_periods_datetime_list():
	return pd.date_range('00:00:00', periods=96, freq='15T').time

# 轉置用電資料
def transpose_data_electricity_watt(date_df):
	period_index = 0
	df_list = []

	# 若已有 96 筆，就可以不用補植
	if (len(date_df) == 96):
		return date_df.drop(['Reporttime'], axis=1)['w'].tolist()
    # 若未有 96 筆，就必須將缺的 period 補成 96 筆
	else:
		for index, row in date_df.iterrows():
			periods = create_periods_datetime_list()
			# 直到找到該時段的 index
			while (row['Reporttime'].time() != periods[period_index]):
				df_list.append(None)
				period_index += 1

			# print(period_index, row['Reporttime'].time(), periods[period_index], row['Reporttime'].time() == periods[period_index])
			df_list.append(row['w'])
			period_index += 1

		# 將最後面幾個 period 的 NA 值都設為 None
		if (len(df_list) != 96):
			df_list.append(None)

			while (len(df_list) != 96):
				df_list.append(None)
				period_index += 1
	return df_list

# 建立一天的資料集
def set_day_dataSet(uuid, userId, reportTime, date_df):
	# print(" ", Reporttime, len(date_df))
	data_watt_list = transpose_data_electricity_watt(date_df)

	# [UUID, User_ID, Reporttime, 1-th, 2-th, ..., 96-th]
	dataSet_list = [uuid, userId, reportTime] + data_watt_list

	# if (len(dataSet_list) != 99):
	# 	print(dataSet_list[2], len(dataSet_list))
	return dataSet_list

# 生成 UUID
# e.g. User_ID: 1, Reporttime: '20180815' -> '1_201808015'
def generate_uuid(userId, reportTime):
	return '{}_{}'.format(userId, reportTime)

# 彙整與轉置單一用戶的用電資料 (96 期)
def consolidation_userId_dataSet(user_dates_group, user_group_name):
	dataSet_lists = []

	# date_group_name (index)：單一用戶一天之時間，date_group (value)：單一用戶一天的用電資料
	for date_group_name, date_group in user_dates_group:
		date_df = date_group.reset_index()
		date_df = date_df.drop(['index', 'User_ID'], axis=1)

		userId = user_group_name
		# 將時間格式 '2018/08/15' 轉換成 '20180815'
		reportTime = date_group_name.strftime('%Y%m%d')
		uuid = generate_uuid(userId, reportTime)

		# 時間只取 年 月 日
		reportTime = date_group_name.date()
		# 建立一天的資料集
		dataSet_list = set_day_dataSet(uuid, userId, reportTime, date_df)
		dataSet_lists.append(dataSet_list)

	return dataSet_lists

# 彙整與轉置多個使用者的用電資料 (96 期)
def consolidation_all_dataSet(dataSet):
	users_group = dataSet.groupby('User_ID')
	users_dataSet_list = []

	# user_group_name (index)：單一用戶編號，user_group (value)：單一用戶用電資料
	for user_group_name, user_group in users_group:
		# 單一用戶以 Reporttime 欄位的每一天 groupby
		user_dates_group = user_group.groupby(pd.Grouper(key='Reporttime', freq='1D'))
		# 彙整與轉置單一用戶的用電資料 (96 期)
		tmp_list = consolidation_userId_dataSet(user_dates_group, user_group_name)
		users_dataSet_list += tmp_list
		# print('process User_ID{}\t{}'.format(user_group_name, len(users_dataSet_list)))

	return users_dataSet_list

# 刪除最前或最後有缺值之資料
def delete_first_or_last_na(dataSet):
	return dataSet.dropna(subset=['1-th', '96-th'])

# 刪除缺值之門檻值
def dorpna_threshold(dataSet, threshold):
	period_sum = 96
	# UUID, User_ID, Reporttime
	another_column_sum = 3
	return dataSet.dropna(thresh=(period_sum - threshold + another_column_sum))
# 	return dataSet.dropna(thresh=(11 - threshold + 1))

# 將 10-th 轉成 10
def transforma_period_list_number(na_periods_colume):
	return [int(period.replace('-th', '')) for period in na_periods_colume]

# period 補值
def fill_period_na(row, na_periods, current_idx):
	current_period = na_periods['colume'][current_idx]
	na_periods['current'] = na_periods['list'][current_idx]
	prev_period = str(na_periods['current'] - 1) + '-th'
	next_period = str(na_periods['current'] + 1) + '-th'

	period_ave = (row[next_period] + row[prev_period]) / 2

	# print('{}: {}, p{}_{}, n{}_{} = c{}-{}'.format(
	# 	row['UUID'], 'fill',
	# 	(na_periods['current'] - 1), row[prev_period],
	# 	(na_periods['current'] + 1), row[next_period],
	# 	na_periods['current'], round(period_ave, 2)))

	row[current_period] = round(period_ave, 2)

# period 缺值處理
def process_period_na(row):
	# print('\n' + '=' * 40)
	na_periods = {}
	na_periods['colume'] = row.index[row.isnull()].tolist()
	na_periods['len'] = len(na_periods['colume'])

	if (na_periods['len'] == 0):
		# print('o', row['UUID'], na_periods['len'])
		return row
	else:
		na_periods['list'] = transforma_period_list_number(na_periods['colume'])
		# print('x', row['UUID'], na_periods['len'], na_periods['colume'])

		if (na_periods['len'] == 1):
			fill_period_na(row, na_periods, 0)
		else:
			for idx in range(na_periods['len'] - 1):
				na_periods['current'] = na_periods['list'][idx]
				na_periods['next'] = na_periods['list'][idx + 1]
				# print(na_periods['current'], na_periods['next'])

				if (na_periods['next'] - na_periods['current'] == 1):
					row[row['UUID'] != row['UUID']]
					# print(row['UUID'], 'drop', na_periods['list'])
					# print('drop row', row['UUID'], row.name)
					return np.nan
				else:
					fill_period_na(row, na_periods, idx)

				# print('-' * 40)
			# print(na_periods['list'][na_periods['len'] - 1])

			# 補最後一個 na 的欄位
			fill_period_na(row, na_periods, na_periods['len'] - 1)
# 	print('fill row', row['UUID'], type(row['UUID']), row.name)
	return row

def process_na(dataSet, peroid_column, threshold):
	delete_before_count = len(dataSet)
	dataSet = dorpna_threshold(dataSet, threshold=threshold)
	#print('刪除未達門檻值之資料，before: {}, after: {}'.format(delete_before_count, len(dataSet)))
	print('==== Filer the data under the threshold {} and the rest rate is {:.2f} % ===='.format(threshold, len(dataSet)*100/delete_before_count))

	delete_before_count = len(dataSet)
	dataSet = delete_first_or_last_na(dataSet)
	#print('刪除最前或最後有缺值之資料，before: {}, after: {}'.format(delete_before_count, len(dataSet)))
	print('==== Filter the missing data and the rest rate is {:.2f} % ===='.format(len(dataSet)*100/delete_before_count))


	delete_before_count = len(dataSet)
    # process_period_na function：補值
	dataSet = dataSet.apply(process_period_na, axis=1)
	# 刪除所有欄位都是 na 的幾列
	dataSet = dataSet.dropna(how='all')

	# 轉型別
	dataSet['UUID'] = dataSet['UUID']
	dataSet['User_ID'] = dataSet['User_ID'].astype(int)
	dataSet = transform_time(dataSet, column='Reporttime', format='%Y-%m-%d')

	#print('刪除無法補值之資料，before: {}, after: {}'.format(delete_before_count, len(dataSet)))
	print('==== Filter the data that cannot be dealed with and the rest rate is {:.2f} % ===='.format(len(dataSet)*100/delete_before_count))
	return dataSet

# 把需量轉成小時 (每 4 個 period，會組成一個 tuple)
def kWh_group(sequence, chunk_size):
	return list(zip(*[iter(sequence)] * chunk_size))

# 計算 每小時 (4 個 period) 平均用電
def hour_mean_w(period_group, dataSet):
	tmp_dataSet = dataSet.loc[:, period_group]
	tmp_list = tmp_dataSet.agg('mean', axis=1)
	return tmp_list

# 計算 最大需量、最大需量、總用電量
def peroid_max_min_sum_w(dataSet):
	peroid_column = create_peroid_column()
	# 只取所有 period 欄位來計算 max 和 min
	tmp_dataSet = dataSet.loc[:, peroid_column]
	dataSet['Max_load'] = tmp_dataSet.agg('max', axis=1)
	dataSet['Min_load'] = tmp_dataSet.agg('min', axis=1)

	# 把需量轉成小時 (每 4 個 period，會組成一個 tuple)
	period_groups = kWh_group(peroid_column, 4)
	# 計算 每小時 (4 個 period) 平均用電
	tmp_mean_dataSet = pd.DataFrame(list(map(lambda x: hour_mean_w(x, tmp_dataSet), period_groups))).T
	# 下面 3 行是原本 list(map(lambda x: hour_mean_w(x, tmp_dataSet), period_groups)) 的詳細版
	# tmp_list = []
	# for kWh in period_groups:
	# 	tmp_list.append(m.hour_mean_w(kWh, tmp_dataSet))	# append Series

	# 計算 總用電量
	dataSet['Total_load'] = tmp_mean_dataSet.agg('sum', axis=1)
	# 將 最大需量、最大需量、總用電量 都做四捨五入至小數第二位
	dataSet = dataSet.round({'Max_load': 2, 'Min_load': 2, 'Total_load': 2})
	return dataSet
