#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import pickle
from lib.util.source import data_extractor
import configparser
import math

DIR_FILE = '../data/config/dir_path.ini'
DIR_CONFIG = configparser.ConfigParser()
DIR_CONFIG.read(DIR_FILE)
DATA_PATH = DIR_CONFIG['DIR']['DATA_DIR']

class dataset:
    def __init__(self, pkl_file=None, cluster_file='user_group_relation.csv', BI_file='user_info.csv',
                 tax_file='city_tax.csv', weather_file='city_weather.csv', test_users=None, days=30,
                 mode='train', enc=None, standardize=False, min_max=False):
        if pkl_file != None:
            path = DATA_PATH + '/result/dataset/'
            self.filename = path + pkl_file + '.pkl'
            self.load_pkl()
            return

        self.input = {}
        self.concat_input = []
        self.batch_idx = 0
        self.data_size = 0
        self.is_concat = False
        self.normalized = False
        self.mode = mode
        self.enc = enc

        self.init_datasource(cluster_file, BI_file, tax_file, weather_file, test_users, days, mode, standardize, min_max)

    def init_datasource(self, cluster_file, BI_file, tax_file, weather_file, test_users, days, mode, standardize, min_max):
        data_source = data_extractor(cluster_file, BI_file, tax_file, weather_file, mode=mode)
        data = data_source.init_with_csv(test_users, days, standardize, min_max)
        self.input['BI'] = data['BI']
        self.input['tax'] = data['tax']
        self.input['weather'] = data['weather']
        self.label = data['group_label']
        self.BI_size = data['BI_size']
        self.tax_size = data['tax_size']
        self.weather_size = data['weather_size']
        self.data_size = len(self.label)

        if len(self.input['BI']) != self.data_size or len(self.input['tax']) != self.data_size or len(self.input['weather']) != self.data_size:
            raise ValueError('Illegal dataset size!')

    def num_examples(self):
        return self.data_size

    def get_dataprocessor(self):
        return self.enc

    def concat_input_data(self):
        if len(self.concat_input) == 0:
            for idx in range(len(self.input['BI'])):
                concat_data = np.concatenate([self.input['BI'][idx], self.input['tax'][idx], self.input['weather'][idx]], axis=0)
                self.concat_input.append(concat_data)
        return self.concat_input

    def generate_naive_dataset(self, concat=True, one_hot=False):
        if self.data_size == 0:
            raise ValueError('Empty dataset!')

        if one_hot:
            if self.enc == None and self.mode == 'train':
                self.enc = OneHotEncoder(handle_unknown='ignore')
                self.naive_y = self.enc.fit_transform(np.array(self.label).reshape(-1, 1)).toarray()
            else:
                self.naive_y = self.enc.transform(np.array(self.label).reshape(-1, 1)).toarray()
        else:
            self.naive_y = self.label

        if not concat:
            self.naive_X = []
            for idx in range(len(self.input['BI'])):
                self.naive_X.append([self.input['BI'][idx], self.input['tax'][idx], self.input['weather'][idx]])
            if self.mode == 'test':
                self.naive_X = self.input
        else:
            self.naive_X = self.concat_input_data()
            self.is_concat = True

    def split_dataset(self, test_size=0.3, concat=True, one_hot=False):
        if len(self.input['BI']) != self.data_size or len(self.input['tax']) != self.data_size or len(self.input['weather']) != self.data_size:
            raise ValueError('Illegal dataset size!')

        self.generate_naive_dataset(concat, one_hot)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.naive_X, self.naive_y,
                                                                                test_size=test_size, random_state=0)
        self.is_concat = concat

    def next_train_batch(self, batch_size):
        self.batch_idx = self.batch_idx + batch_size
        if self.y_train is None or self.X_train is None:
            print('Empty training dataset!')
            return

        if self.batch_idx > len(self.X_train):
            self.batch_idx = batch_size

        batch_x = self.X_train[self.batch_idx - batch_size:self.batch_idx]
        batch_y = self.y_train[self.batch_idx - batch_size:self.batch_idx]

        if self.is_concat:
            return batch_x, batch_y
        else:
            return self.get_non_concate_dataset(batch_x, batch_y)

    def get_non_concate_dataset(self, input_x, label_y):
        datasize = len(input_x)
        if len(label_y) != datasize:
            raise ValueError('Invalid dataset!')

        x_BI = []
        x_tax = []
        x_weather = []
        for idx in range(datasize):
            x_BI.append(input_x[idx][0])
            x_tax.append(input_x[idx][1])
            x_weather.append(input_x[idx][2])
        return {'BI': x_BI, 'tax': x_tax, 'weather': x_weather}, label_y

    def load_pkl(self):
        file = open(self.filename, 'rb')
        tmp_dict = pickle.load(file)
        file.close()
        self.__dict__.update(tmp_dict)

    def save_pkl(self, file_name):
        file_path = DATA_PATH + '/result/dataset/' + file_name + '.pkl'
        pkl_file = open(file_path, 'wb')
        pickle.dump(self.__dict__, pkl_file, 2)
        pkl_file.close()

    def convert_to_binary_label(self, target_label):
        for idx, label in enumerate(self.label):
            if label == target_label:
                self.label[idx] = 1
            else:
                self.label[idx] = 0

    def convert_to_unique_dataset(self, target_label, concat=True, one_hot=False, is_split=False, test_size=0.3):
        self.generate_naive_dataset(concat, one_hot)
        unique_dataset = []
        unique_label = []
        for idx, label in enumerate(self.naive_y):
            if label == target_label:
                unique_dataset.append(self.naive_X[idx])
                unique_label.append(label)

        self.naive_X = np.array(unique_dataset)
        self.naive_y = np.array(unique_label)
        if is_split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.naive_X, self.naive_y,
                                                                                    test_size=test_size, random_state=0)


def generate_test_users(num_extract, file_name='user_group_relation'):
    file_path = DATA_PATH + file_name + '.csv'
    label_df = pd.read_csv(file_path)

    group_label = label_df.groupby('Group_ID')
    test_users = []
    for group_id in group_label.groups.keys():
        users = group_label.get_group(group_id).groupby('User_ID').count().Group_ID.sort_values(ascending=False)[:num_extract]
        test_users.extend(users.index.values)
    return test_users
