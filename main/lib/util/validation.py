#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import configparser
import datetime

DIR_FILE = '../data/config/dir_path.ini'
DIR_CONFIG = configparser.ConfigParser()
DIR_CONFIG.read(DIR_FILE)
DATA_PATH = DIR_CONFIG['DIR']['DATA_DIR']

class metrics:
    def __init__(self, cluster_file='group_center.csv', load_file='for_clustering.csv'):
        self.group_center = pd.read_csv(DATA_PATH + cluster_file)
        self.household_consumption = pd.read_csv(DATA_PATH + load_file)
        self.kwh_factor = 15 * 60 / (1000 * 3600)

    def mean_absolute_error(self, prediction, test_users):
        absolute_error = 0
        num_estimation = prediction.shape[0]

        user_consumption = self.household_consumption.groupby('User_ID')
        group_consumption = self.group_center.sum(axis=1) / self.kwh_factor
        real_consumption = []
        for user in test_users:
            household = user_consumption.get_group(user)
            mean_consumption = household.iloc[:, 3:99].sum(axis=1).mean()
            real_consumption.append(mean_consumption)

        real_consumption = np.array(real_consumption).mean() / self.kwh_factor
        for label in prediction:
            predict_consumption = group_consumption[label]
            absolute_error = absolute_error + abs(predict_consumption-real_consumption)

        mae = absolute_error/num_estimation
        return mae
