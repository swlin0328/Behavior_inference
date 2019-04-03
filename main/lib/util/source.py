#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from ..util.preprocess import min_max, normalize, standardization, median_filter
from ..config.config import DATA_PATH, WEATHER_ATTR, TAX_ATTR, ATTR_CONFIG

class data_extractor:
    def __init__(self, config, cluster_file='user_group_relation.csv', user_file='user_info.csv',
                 tax_file='city_tax.csv', weather_file='city_weather.csv', mode='train'):
        self.config = config
        self.mode = mode
        self.cluster_path = DATA_PATH + cluster_file
        self.user_path = DATA_PATH + user_file
        self.tax_path = DATA_PATH + tax_file
        self.weather_path = DATA_PATH + weather_file

        self.split_dataset_by_user = True
        self.user_size = 0
        self.tax_size = 0
        self.weather_size = 0

        self.user_max = {'min': [],
                       'min_max': []}
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

        self.input_user = []
        self.input_tax = []
        self.input_weather = []
        self.label = []
        self.group_dict = {'Group': {},
                           'Group_key': {}}
        self.load_statistics_result()

    def init_with_csv(self, test_users=None, days=30, is_standardize=False, is_min_max=False):
        if test_users is None or len(test_users) == 0:
            self.split_dataset_by_user = False

        self.extract_cluster_df()
        self.extract_user_df(test_users)
        self.extract_tax_df(is_standardize, is_min_max)
        self.extract_weather_df(is_standardize, is_min_max, days)

        self.generate_dataset_with_csv()
        self.save_statistics_result()

        data_source = {'user': self.input_user, 'tax': self.input_tax, 'weather': self.input_weather, 'group_label': self.label,
                       'user_size': self.user_size, 'tax_size': self.tax_size, 'weather_size': self.weather_size}
        return data_source

    def extract_cluster_df(self):
        cluster_df = pd.read_csv(self.cluster_path)
        self.get_group_and_key(cluster_df, 'cluster', 'User_ID')

    def extract_user_df(self, test_users):
        drop_cols = ['User_ID', 'Location_City', 'Location_Area']
        user_df = pd.read_csv(self.user_path)
        user_df = user_df.set_index(user_df['User_ID'])
        user_df = normalize(user_df, drop_cols, self.user_max)
        if self.split_dataset_by_user:
            user_df = self.select_dataset_by_user(user_df, test_users)
        self.get_group_and_key(user_df, 'user', 'User_ID')

    def extract_tax_df(self, is_standardize, is_min_max):
        tax_df = pd.read_csv(self.tax_path)
        self.get_group_and_key(tax_df, 'tax', '鄉鎮市區')
        if is_standardize:
            self.group_dict['Group']['tax'] = self.group_dict['Group']['tax'].apply(
                lambda x: standardization(x, TAX_ATTR, self.tax_mean, self.tax_std)).groupby(['鄉鎮市區'])
        if is_min_max:
            self.group_dict['Group']['tax'] = self.group_dict['Group']['tax'].apply(
                lambda x: min_max(x, TAX_ATTR, self.tax_max)).groupby(['鄉鎮市區'])

    def extract_weather_df(self, is_standardize, is_min_max, days):
        weather_df = pd.read_csv(self.weather_path)
        weather_df = weather_df.set_index(pd.to_datetime(weather_df['Reporttime']))

        weather_df['City'] = weather_df['Area'].str.split('-', expand=True)[0]
        weather_df['Region'] = weather_df['Area'].str.split('-', expand=True)[1]
        weather_df = self.preprocess_in_duration(weather_df, ATTR_CONFIG['weather'])
        if not self.split_dataset_by_user:
            weather_df = self.select_dataset_by_time(weather_df, days)

        self.get_group_and_key(weather_df, 'weather_city', 'City')
        self.get_group_and_key(weather_df, 'weather_region', 'Region')
        if is_standardize:
            self.group_dict['Group']['weather_city'] = self.group_dict['Group']['weather_city'].apply(
                lambda x: standardization(x, WEATHER_ATTR, self.weather_mean['City'], self.weather_std['City']
                                               )).groupby(['City'])
            self.group_dict['Group']['weather_region'] = self.group_dict['Group']['weather_region'].apply(
                lambda x: standardization(x, WEATHER_ATTR, self.weather_mean['Region'], self.weather_std['Region']
                                               )).groupby(['Region'])
        if is_min_max:
            self.group_dict['Group']['weather_city'] = self.group_dict['Group']['weather_city'].apply(
                lambda x: min_max(x, WEATHER_ATTR, self.weather_max['City'])).groupby(['City'])
            self.group_dict['Group']['weather_region'] = self.group_dict['Group']['weather_region'].apply(
                lambda x: min_max(x, WEATHER_ATTR, self.weather_max['Region'])).groupby(['Region'])

    def preprocess_in_duration(self, df, duration):
        format_str = '%Y/%m/%d'
        start_date = datetime.datetime.strptime(duration['start_date'], format_str).date()
        end_date = datetime.datetime.strptime(duration['end_date'], format_str).date()
        duration = np.logical_and(df.index.date > start_date, df.index.date < end_date)
        df = df[duration].groupby(['Region'])
        df = df.apply(lambda x: median_filter(x, 3, WEATHER_ATTR))
        return df

    def select_dataset_by_time(self, weather_df, days):
        split_date = datetime.datetime.today().date() - datetime.timedelta(days)
        if self.mode == 'train':
            weather_df = weather_df[weather_df.index.date < split_date]
        else:
            weather_df = weather_df[weather_df.index.date > split_date]
        return weather_df

    def select_dataset_by_user(self, user_df, test_users):
        if self.mode == 'train':
            user_df = user_df[[idx not in test_users for idx in user_df.index]]
        else:
            user_df = user_df[[idx in test_users for idx in user_df.index]]
        return user_df

    def generate_dataset_with_csv(self):
        for user in tqdm(self.group_dict['Group_key']['user'], ncols=60):
            if user not in self.group_dict['Group_key']['cluster']:
                continue

            user_data, tax_data, location = self.extract_user_data(user)
            location['region'] = location['region'].strip('區')
            self.group_dict['Group']['daily_group'] = self.group_dict['Group']['cluster'].get_group(user).groupby('Week_ID')
            self.group_dict['Group_key']['daily_group'] = self.group_dict['Group']['daily_group'].groups.keys()
            self.combine_user_data_with_daily_weather(user_data, tax_data, location)

    def extract_user_data(self, user):
        target_user = self.group_dict['Group']['user'].get_group(user)
        city = target_user.values[0][1]
        region = target_user.values[0][2]

        taget_region_tax = self.group_dict['Group']['tax'].get_group(region)
        taget_region_tax = taget_region_tax.set_index(['村里'])

        user_data = target_user.values[0][4:]
        tax_data = taget_region_tax.loc['合計', TAX_ATTR].values.reshape([-1], order='F')

        if self.user_size == 0:
            self.user_size = user_data.size
        if self.tax_size == 0:
            self.tax_size = tax_data.size
        return user_data, tax_data, {'city': city, 'region': region}

    def combine_user_data_with_daily_weather(self, user_data, tax_data, location):
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
                weather_data = weather_df[weather_df.index.date == date_idx][WEATHER_ATTR].values.reshape([-1],
                                                                                                          order='F')
                if self.weather_size == 0:
                    self.weather_size = weather_data.size
                if not self.is_valid_data(user_data, tax_data, weather_data):
                    continue

                group_label = self.group_dict['Group']['daily_group'].get_group(weekid)['Group_ID'].values[0]
                self.input_user.append(user_data)
                self.input_tax.append(tax_data)
                self.input_weather.append(weather_data)
                self.label.append(group_label)

    def compute_average_weather(self, weather_df, location):
        if location not in self.weather_city_avg.keys():
            time_index = weather_df.index.unique()
            avg_csv = pd.DataFrame(index=time_index, columns=WEATHER_ATTR)
            for time in time_index:
                if weather_df.loc[time, WEATHER_ATTR].size > 1:
                    avg_csv.loc[time, WEATHER_ATTR] = weather_df.loc[time, WEATHER_ATTR].mean(axis=0)
                else:
                    avg_csv.loc[time, WEATHER_ATTR] = weather_df.loc[time, WEATHER_ATTR]
            self.weather_city_avg[location] = avg_csv

        return self.weather_city_avg[location]

    def is_valid_data(self, user_data, tax_data, weather_data):
        if user_data.size != self.user_size:
            return False
        if tax_data.size != self.tax_size:
            return False
        if weather_data.size != self.weather_size:
            return False

        if pd.isnull(user_data).any():
            return False
        if pd.isnull(tax_data).any():
            return False
        if pd.isnull(weather_data).any():
            return False
        return True

    def get_group_and_key(self, df, dict_key, group_key):
        self.group_dict['Group'][dict_key] = df.groupby(group_key)
        self.group_dict['Group_key'][dict_key] = self.group_dict['Group'][dict_key].groups.keys()

    def save_statistics_result(self, path=None):
        if self.mode == 'train':
            if path is None:
                path = DATA_PATH + '/result/parameters/' + self.config['model_name'] + '_statistics.npz'
            tax_array = [self.tax_max, self.tax_mean, self.tax_std]
            weather_array = [self.weather_max, self.weather_mean, self.weather_std]
            user_array = [self.user_max]
            np.savez(path, tax_array=tax_array, weather_array=weather_array, user_array=user_array)

    def load_statistics_result(self, path=None):
        if self.mode == 'test':
            if path is None:
                    path = DATA_PATH + '/result/parameters/' + self.config['model_name'] + '_statistics.npz'

            statistics = np.load(path)
            self.tax_max = statistics['tax_array'][0]
            self.tax_mean = statistics['tax_array'][1]
            self.tax_std = statistics['tax_array'][2]
            self.weather_max = statistics['weather_array'][0]
            self.weather_mean = statistics['weather_array'][1]
            self.weather_std = statistics['weather_array'][2]
            self.user_max = statistics['user_array'][0]
