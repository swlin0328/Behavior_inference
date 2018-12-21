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
                 tax_file='city_tax.csv', weather_file='city_weather.csv'):
        self.cluster_path = DATA_PATH + cluster_file
        self.BI_path = DATA_PATH + BI_file
        self.tax_path = DATA_PATH + tax_file
        self.weather_path = DATA_PATH + weather_file
        self.BI_size = 0
        self.tax_size = 0
        self.weather_size = 0

        self.input_BI = []
        self.input_tax = []
        self.input_weather = []
        self.label = []

    def init_with_csv(self, test_users=None, days=30, mode='train'):
        cluster_df = pd.read_csv(self.cluster_path)
        tax_df = pd.read_csv(self.tax_path)

        BI_df = pd.read_csv(self.BI_path)
        BI_df = BI_df.set_index(BI_df['User_ID'])
        weather_df = pd.read_csv(self.weather_path)
        weather_df = weather_df.set_index(pd.to_datetime(weather_df['Reporttime']))
        weather_df['City'] = weather_df['Area'].str.split('-', expand=True)[0]
        weather_df['Region'] = weather_df['Area'].str.split('-', expand=True)[1]

        if test_users is None or len(test_users)==0:
            weather_df = self.select_dataset_by_time(weather_df, days, mode)
        else:
            BI_df = self.select_dataset_by_user(BI_df, test_users, mode)

        group_dict = self.get_group_and_keys(BI_df, tax_df, weather_df, cluster_df)
        self.generate_dataset_with_csv(group_dict)
        data_source = {'BI':self.input_BI, 'tax':self.input_tax, 'weather':self.input_weather,
                        'group_label':self.label, 'BI_size': self.BI_size, 'tax_size': self.tax_size, 'weather_size': self.weather_size}

        return data_source

    def select_dataset_by_time(self, weather_df, days, mode):
        split_date = datetime.datetime.today().date() - datetime.timedelta(days)
        if mode == 'train':
            weather_df = weather_df[weather_df.index.date < split_date]
        else:
            weather_df = weather_df[weather_df.index.date > split_date]

        return weather_df

    def select_dataset_by_user(self, BI_df, test_users, mode):
        if mode == 'train':
            BI_df = BI_df[[idx not in test_users for idx in BI_df.index]]
        else:
            BI_df = BI_df[[idx in test_users for idx in BI_df.index]]

        return BI_df

    def generate_dataset_with_csv(self, group_dict):
        for user in tqdm(group_dict['Group_key']['BI'], ncols=60):
            if user not in group_dict['Group_key']['cluster']:
                continue

            BI_data, tax_data, location = self.extract_user_data(group_dict, user)

            group_dict['Group']['daily_group'] = group_dict['Group']['cluster'].get_group(user).groupby('Week_ID')
            group_dict['Group_key']['daily_group'] = group_dict['Group']['daily_group'].groups.keys()

            self.combine_user_data_with_daily_weather(BI_data, tax_data, group_dict, location)

    def extract_user_data(self, group_dict, user):
        target_user = group_dict['Group']['BI'].get_group(user)
        city = target_user.values[0][1]
        region = target_user.values[0][2]

        taget_region_tax = group_dict['Group']['tax'].get_group(region)
        taget_region_tax = taget_region_tax.set_index(['村里'])

        BI_data = target_user.values[0][3:]
        tax_data = taget_region_tax.loc['合計', tax_attr].values.reshape(-1)

        if self.BI_size == 0:
            self.BI_size = BI_data.size

        if self.tax_size == 0:
            self.tax_size = tax_data.size

        return BI_data, tax_data, {'city': city, 'region': region}

    def combine_user_data_with_daily_weather(self, BI_data, tax_data, group_dict, location):
        if location['region'] in group_dict['Group_key']['weather_region']:
            weather_df = group_dict['Group']['weather_region'].get_group(location['region'])
        elif location['city'] in group_dict['Group_key']['weather_city']:
            weather_df = group_dict['Group']['weather_city'].get_group(location['city'])
        else:
            return

        for weekid in group_dict['Group_key']['daily_group']:
            weather_for_weekid = weather_df[(weather_df.index.weekday + 1) == weekid]
            date = pd.unique(weather_for_weekid.index.date)
            for date_idx in date:
                weather_data = weather_df[weather_df.index.date == date_idx][weather_attr].values.reshape(-1)
                if self.weather_size == 0:
                    self.weather_size = weather_data.size
                if not self.is_valid_size(BI_data, tax_data, weather_data):
                    continue

                group_label = group_dict['Group']['daily_group'].get_group(weekid)['Group_ID'].values[0]
                self.input_BI.append(BI_data)
                self.input_tax.append(tax_data)
                self.input_weather.append(weather_data)
                self.label.append(group_label)

    def is_valid_size(self, BI_data, tax_data, weather_data):
        if BI_data.size != self.BI_size:
            return False
        if tax_data.size != self.tax_size:
            return False
        if weather_data.size != self.weather_size:
            return False
        return True

    def get_group_and_keys(self, BI_df, tax_df, weather_df, cluster_df):
        BI_group = BI_df.groupby('User_ID')
        BI_group_keys = BI_group.groups.keys()

        tax_group = tax_df.groupby('鄉鎮市區')
        tax_group_keys = tax_group.groups.keys()

        weather_city_group = weather_df.groupby('City')
        weather_city_group_keys = weather_city_group.groups.keys()

        weather_region_group = weather_df.groupby('Region')
        weather_region_group_keys = weather_city_group.groups.keys()

        cluster_group = cluster_df.groupby('User_ID')
        cluster_group_keys = cluster_group.groups.keys()

        group_dict = {'BI': BI_group, 'tax': tax_group, 'weather_city': weather_city_group,
                      'weather_region': weather_region_group,
                      'cluster': cluster_group}
        group_key_dict = {'BI': BI_group_keys, 'tax': tax_group_keys, 'weather_city': weather_city_group_keys,
                          'weather_region': weather_region_group_keys,
                          'cluster': cluster_group_keys}

        return {'Group': group_dict, 'Group_key': group_key_dict}