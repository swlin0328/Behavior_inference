# coding: utf-8

import _pickle as cPickle
from time import strftime
import os
from ..cluster.preprocess import dataset_preprocessing
from ..cluster.user_dailyload_cal import user_dailyload_cal
from ..cluster.user_dailyload_to_group import user_dailyload_to_group
from ..db.raw_dataset import sql4data


class clustering():
	def __init__(self, model_name, sql_conn=None):
		self.model_name = model_name
		self.sql_conn = sql_conn
		#sql4data().start(start_date='', end_date='', sql_conn=sql_conn)

	def generate_group_dailyload(self):
		preprocess_dataSet = dataset_preprocessing.start()
		user_dailyload_df = user_dailyload_cal(preprocess_dataSet).start()
		user_dailyload_group = user_dailyload_to_group(user_dailyload_df, self.model_name, self.sql_conn).start()

	def start(self):
		self.generate_group_dailyload()
