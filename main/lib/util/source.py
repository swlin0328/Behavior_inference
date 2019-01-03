#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import configparser
import datetime
import ast
from tqdm import tqdm

DIR_FILE = '../data/config/dir_path.ini'
DIR_CONFIG = configparser.ConfigParser()
DIR_CONFIG.read(DIR_FILE)
DATA_PATH = DIR_CONFIG['DIR']['DATA_DIR']

ATTR_FILE = '../data/config/infer_attr.ini'
ATTR_CONFIG = configparser.ConfigParser()
ATTR_CONFIG.read(ATTR_FILE)
weather_attr = ast.literal_eval(ATTR_CONFIG['weather']['attr'])
tax_attr = ast.literal_eval(ATTR_CONFIG['tax']['attr'])

class data_extractor:
    def __init__(self, cluster_file='user_group_relation.csv', BI_file='user_info.csv',
                 tax_file='city_tax.csv', weather_file='city_weather.csv', mode='train'):
        self.mode = mode
        self.cluster_path = DATA_PATH + cluster_file
        self.BI_path = DATA_PATH + BI_file
        self.tax_path = DATA_PATH + tax_file
        self.weather_path = DATA_PATH + weather_file

        self.split_dataset_by_user = True
        self.BI_size = 0
        self.tax_size = 0
        self.weather_size = 0

        self.tax_max = {}
        self.tax_mean = {}
        self.tax_std = {}
        self.weather_max = {'City': {},
                            'Region': {}}
        self.weather_mean = {'City': {},
                             'Region': {}}
        self.weather_std = {'City': {},
                            'Region': {}}
        self.weather_city_avg = {}

        self.input_BI = []
        self.input_tax = []
        self.input_weather = []
        self.label = []
        self.group_dict = {'Group': {},
                           'Group_key': {}}

        self.load_statistics_result()

    def init_with_csv(self, test_users=None, days=30, standardize=False, Min_Max=False):
        if test_users is None or len(test_users) == 0:
            self.split_dataset_by_user = False

        self.extract_cluster_df()
        self.extract_BI_df(test_users)
        self.extract_tax_df(standardize, Min_Max)
        self.extract_weather_df(standardize, Min_Max, days)

        self.generate_dataset_with_csv()
        self.save_statistics_result()

        data_source = {'BI': self.input_BI, 'tax': self.input_tax, 'weather': self.input_weather, 'group_label': self.label,
                       'BI_size': self.BI_size, 'tax_size': self.tax_size, 'weather_size': self.weather_size}
        return data_source

    def extract_cluster_df(self):
        cluster_df = pd.read_csv(self.cluster_path)
        self.get_group_and_key(cluster_df, 'cluster', 'User_ID')

    def extract_BI_df(self, test_users):
        drop_cols = ['User_ID', 'City', 'Region']
        BI_df = pd.read_csv(self.BI_path)
        BI_df = BI_df.set_index(BI_df['User_ID'])
        BI_df = self.normalize(BI_df, drop_cols)
        if self.split_dataset_by_user:
            BI_df = self.select_dataset_by_user(BI_df, test_users)
        self.get_group_and_key(BI_df, 'BI', 'User_ID')

    def extract_tax_df(self, standardize, Min_Max):
        tax_df = pd.read_csv(self.tax_path)
        self.get_group_and_key(tax_df, 'tax', '鄉鎮市區')
        if standardize:
            self.group_dict['Group']['tax'] = self.group_dict['Group']['tax'].apply(
                lambda x: self.standardization(x, tax_attr, self.tax_mean, self.tax_std)).groupby(['鄉鎮市區'])
        if Min_Max:
            self.group_dict['Group']['tax'] = self.group_dict['Group']['tax'].apply(
                lambda x: self.min_max(x, tax_attr, self.tax_max)).groupby(['鄉鎮市區'])

    def extract_weather_df(self, standardize, Min_Max, days):
        weather_df = pd.read_csv(self.weather_path)
        weather_df = weather_df.set_index(pd.to_datetime(weather_df['Reporttime']))
        weather_df['City'] = weather_df['Area'].str.split('-', expand=True)[0]
        weather_df['Region'] = weather_df['Area'].str.split('-', expand=True)[1]
        if not self.split_dataset_by_user:
            weather_df = self.select_dataset_by_time(weather_df, days)

        self.get_group_and_key(weather_df, 'weather_city', 'City')
        self.get_group_and_key(weather_df, 'weather_region', 'Region')
        if standardize:
            self.group_dict['Group']['weather_city'] = self.group_dict['Group']['weather_city'].apply(
                lambda x: self.standardization(x, weather_attr, self.weather_mean['City'], self.weather_std['City']
                                               )).groupby(['City'])
            self.group_dict['Group']['weather_region'] = self.group_dict['Group']['weather_region'].apply(
                lambda x: self.standardization(x, weather_attr, self.weather_mean['Region'], self.weather_std['Region']
                                               )).groupby(['Region'])
        if Min_Max:
            self.group_dict['Group']['weather_city'] = self.group_dict['Group']['weather_city'].apply(
                lambda x: self.min_max(x, weather_attr, self.weather_max['City'])).groupby(['City'])
            self.group_dict['Group']['weather_region'] = self.group_dict['Group']['weather_region'].apply(
                lambda x: self.min_max(x, weather_attr, self.weather_max['Region'])).groupby(['Region'])

    def select_dataset_by_time(self, weather_df, days):
        split_date = datetime.datetime.today().date() - datetime.timedelta(days)
        if self.mode == 'train':
            weather_df = weather_df[weather_df.index.date < split_date]
        else:
            weather_df = weather_df[weather_df.index.date > split_date]
        return weather_df

    def select_dataset_by_user(self, BI_df, test_users):
        if self.mode == 'train':
            BI_df = BI_df[[idx not in test_users for idx in BI_df.index]]
        else:
            BI_df = BI_df[[idx in test_users for idx in BI_df.index]]
        return BI_df

    def generate_dataset_with_csv(self):
        for user in tqdm(self.group_dict['Group_key']['BI'], ncols=60):
            if user not in self.group_dict['Group_key']['cluster']:
                continue

            BI_data, tax_data, location = self.extract_user_data(user)
            self.group_dict['Group']['daily_group'] = self.group_dict['Group']['cluster'].get_group(user).groupby('Week_ID')
            self.group_dict['Group_key']['daily_group'] = self.group_dict['Group']['daily_group'].groups.keys()
            self.combine_user_data_with_daily_weather(BI_data, tax_data, location)

    def extract_user_data(self, user):
        target_user = self.group_dict['Group']['BI'].get_group(user)
        city = target_user.values[0][1]
        region = target_user.values[0][2]

        taget_region_tax = self.group_dict['Group']['tax'].get_group(region)
        taget_region_tax = taget_region_tax.set_index(['村里'])

        BI_data = target_user.values[0][3:]
        tax_data = taget_region_tax.loc['合計', tax_attr].values.reshape(-1)

        if self.BI_size == 0:
            self.BI_size = BI_data.size
        if self.tax_size == 0:
            self.tax_size = tax_data.size
        return BI_data, tax_data, {'city': city, 'region': region}

    def combine_user_data_with_daily_weather(self, BI_data, tax_data, location):
        if location['region'] in self.group_dict['Group_key']['weather_region']:
            weather_df = self.group_dict['Group']['weather_region'].get_group(location['region'])
        elif location['city'] in self.group_dict['Group_key']['weather_city']:
            weather_city = self.group_dict['Group']['weather_city'].get_group(location['city'])
            weather_df = self.compute_average_weather(weather_city, location['city'])
        else:
            return

        for weekid in self.group_dict['Group_key']['daily_group']:
            weather_for_weekid = weather_df[(weather_df.index.weekday + 1) == weekid]
            date = pd.unique(weather_for_weekid.index.date)
            for date_idx in date:
                weather_data = weather_df[weather_df.index.date == date_idx][weather_attr].values.reshape(-1)
                if self.weather_size == 0:
                    self.weather_size = weather_data.size
                if not self.is_valid_size(BI_data, tax_data, weather_data):
                    continue

                group_label = self.group_dict['Group']['daily_group'].get_group(weekid)['Group_ID'].values[0]
                self.input_BI.append(BI_data)
                self.input_tax.append(tax_data)
                self.input_weather.append(weather_data)
                self.label.append(group_label)

    def compute_average_weather(self, weather_df, location):
        if location not in self.weather_city_avg.keys():
            time_index = weather_df.index.unique()
            avg_csv = pd.DataFrame(index=time_index, columns=weather_attr)
            for time in time_index:
                if weather_df.loc[time, weather_attr].size > 1:
                    avg_csv.loc[time, weather_attr] = weather_df.loc[time, weather_attr].mean(axis=0)
                else:
                    avg_csv.loc[time, weather_attr] = weather_df.loc[time, weather_attr]
            self.weather_city_avg[location] = avg_csv

        return self.weather_city_avg[location]

    def is_valid_size(self, BI_data, tax_data, weather_data):
        if BI_data.size != self.BI_size:
            return False
        if tax_data.size != self.tax_size:
            return False
        if weather_data.size != self.weather_size:
            return False
        return True

    def get_group_and_key(self, df, dict_key, group_key):
        self.group_dict['Group'][dict_key] = df.groupby(group_key)
        self.group_dict['Group_key'][dict_key] = self.group_dict['Group'][dict_key].groups.keys()

    def min_max(self, df_group, attr, max_dict):
        group_key = df_group.name
        series = df_group.loc[:, attr]
        if group_key not in max_dict.keys():
            max_dict[group_key] = series.max()
        series_normalize = series / max_dict[group_key]
        df_group.loc[:, attr] = series_normalize
        return df_group

    def normalize(self, df, attr):
        for col in df.columns.values:
            if col not in attr:
                series = df.loc[:, col]
                max = series.max()
                series_normalize = series / max
                df.loc[:, col] = series_normalize
        return df

    def standardization(self, df_group, attr, avg_dict, stdv_dict):
        group_key = df_group.name
        series = df_group.loc[:, attr]
        if group_key not in avg_dict.keys() or group_key not in stdv_dict.keys():
            avg_dict[group_key] = series.mean()
            stdv_dict[group_key] = series.std()
        series_standardized = (series - avg_dict[group_key]) / stdv_dict[group_key]
        df_group.loc[:, attr] = series_standardized
        return df_group

    def save_statistics_result(self, path=None):
        if self.mode == 'train':
            if path is None:
                path = DATA_PATH + '/result/parameters/statistics.npz'
            tax_array = [self.tax_max, self.tax_mean, self.tax_std]
            weather_array = [self.weather_max, self.weather_mean, self.weather_std]
            np.savez(path, tax_array=tax_array, weather_array=weather_array)

    def load_statistics_result(self, path=None):
        if self.mode == 'test':
            if path is None:
                path = DATA_PATH + '/result/parameters/statistics.npz'

            statistics = np.load(path)
            self.tax_max = statistics['tax_array'][0]
            self.tax_mean = statistics['tax_array'][1]
            self.tax_std = statistics['tax_array'][2]
            self.weather_max = statistics['weather_array'][0]
            self.weather_mean = statistics['weather_array'][1]
            self.weather_std = statistics['weather_array'][2]