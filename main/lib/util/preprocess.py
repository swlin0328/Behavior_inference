#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import math
import configparser
from ..config.config import DATA_PATH

def min_max(df_group, attr, max_dict):
    group_key = df_group.name
    series = df_group.loc[:, attr]
    series_max = series.max()
    series_min = series.min()
    if group_key not in max_dict.keys():
        min_max = series_max - series_min
        max_dict[group_key] = min_max
    series_normalize = (series - series_min) / max_dict[group_key]
    df_group.loc[:, attr] = series_normalize
    return df_group


def normalize(df, drop_attr, max_dict):
    attr = set(df.columns.values) - set(drop_attr)
    series = df.loc[:, attr]
    series_max = series.max()
    series_min = series.min()
    if len(max_dict['min']) < 1:
        max_dict['min'] = series_min
        max_dict['min_max'] = series_max - series_min
    series_normalize = (series - max_dict['min']) / max_dict['min_max']
    df.loc[:, attr] = series_normalize
    return df


def standardization(df_group, attr, avg_dict, stdv_dict):
    group_key = df_group.name
    series = df_group.loc[:, attr]
    if group_key not in avg_dict.keys() or group_key not in stdv_dict.keys():
        avg_dict[group_key] = series.mean()
        stdv_dict[group_key] = series.std()
    series_standardized = (series - avg_dict[group_key]) / stdv_dict[group_key]
    df_group.loc[:, attr] = series_standardized
    return df_group


def median_filter(df, window_size, attr):
    start_idx = math.floor(window_size / 2)
    end_idx = -1 - math.floor(window_size / 2)
    for col in attr:
        df.iloc[start_idx:end_idx][col] = df.iloc[start_idx:end_idx][col].rolling(window_size, center=True,
                                                                                  min_periods=1).median()
    return df


def drop_features(source_name, target_name, drop_feature):
    source_path = DATA_PATH + source_name + '.csv'
    target_path = DATA_PATH + target_name + '.csv'
    df = pd.read_csv(source_path)
    df = df.drop(columns=drop_feature)
    df.to_csv(target_path, encoding='utf_8_sig', index=False)


def extract_features_name(source_name):
    source_path = DATA_PATH + source_name + '.csv'
    df = pd.read_csv(source_path)
    features = df.columns.values
    return features


def feature_engineering(df, attribute_1, attribute_2, operator):
    new_feature = attribute_1 + operator + attribute_2
    if operator == '*':
        result = df[attribute_1].multiply(df[attribute_2], fill_value=0)
    if operator == '/':
        result = df[attribute_1].divide(df[attribute_2], fill_value=0)

    df[new_feature] = result
    return df, new_feature


def find_location(district):
    path = DATA_PATH + 'country_hashtable.csv'

    map_rule = pd.read_csv(path)
    location_rule = map_rule[['行政區', '中心點緯度', '中心點經度']]
    location_rule = location_rule.set_index('行政區')

    longitude = location_rule.loc[district, '中心點經度']
    latitude = location_rule.loc[district, '中心點緯度']
    location = {'經度':longitude, '緯度':latitude}
    return location


def get_num_group(file_name='user_group_relation'):
    file_path = DATA_PATH + file_name + '.csv'
    label_df = pd.read_csv(file_path)

    group_label = label_df.groupby('Group_ID')
    num_group = len(group_label.groups.keys())
    return num_group

def generate_test_users(num_extract, file_name='user_group_relation'):
    file_path = DATA_PATH + file_name + '.csv'
    label_df = pd.read_csv(file_path)
    num_extract = int(num_extract)

    group_label = label_df.groupby('Group_ID')
    test_users = []
    for group_id in group_label.groups.keys():
        users = group_label.get_group(group_id).groupby('User_ID').count().Group_ID.sort_values(
            ascending=False)[:num_extract]
        test_users.extend(users.index.values)
    return test_users
