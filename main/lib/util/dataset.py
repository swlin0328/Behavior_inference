#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import pickle
from lib.util.source import data_extractor
from sklearn.preprocessing import StandardScaler

class dataset:
    def __init__(self, pkl_file=None, cluster_file='user_group_relation.csv', BI_file='user_info.csv', tax_file='city_tax.csv',
                 weather_file='city_weather.csv', test_users=None, days=30, mode='train', enc=None, scaler = None, standardize = False):
        if pkl_file != None:
            self.filename = pkl_file
            self.load_pkl()
            return

        self.concat_input = []
        self.batch_idx = 0
        self.data_size = 0
        self.is_concat = False
        self.enc = enc
        self.mode = mode
        self.input = {}
        self.scaler = scaler

        self.init_datasource(cluster_file, BI_file, tax_file, weather_file, test_users, days, mode)
        self.init_dataprocessor()
        if standardize:
            self.standardize_dataset()

    def init_dataprocessor(self):
        if self.enc == None and self.mode == 'train':
            self.enc = OneHotEncoder(handle_unknown='ignore')

        if self.scaler == None and self.mode == 'train':
            self.scaler = {}

    def init_datasource(self, cluster_file, BI_file, tax_file, weather_file, test_users, days, mode):
        data_source = data_extractor(cluster_file, BI_file, tax_file, weather_file)
        data = data_source.init_with_csv(test_users, days, mode)
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
        return self.scaler, self.enc

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
            if self.mode == 'train':
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

    def split_dataset(self, test_size=0.3, concat=True, one_hot=False):
        if len(self.input['BI']) != self.data_size or len(self.input['tax']) != self.data_size or len(self.input['weather']) != self.data_size:
            raise ValueError('Illegal dataset size!')

        self.generate_naive_dataset(concat, one_hot)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.naive_X, self.naive_y,
                                                                                test_size=test_size, random_state=0)
        self.is_concat = concat

    def standardize_dataset(self):
        if self.scaler is None:
            raise ValueError('None type of standardizer!')

        if self.mode == 'train':
            for key, data in self.input.items():
                self.scaler[key] = StandardScaler()
                self.scaler[key].fit(data)

        for key, data in self.input.items():
            self.input[key] = self.scaler[key].transform(data)

        return self.scaler

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

    def load_model_from_pkl(self):
        model_path = 'model/' + self.model_name
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def save_model_to_pkl(self, trained_model):
        file_path = r'model/' + self.model_name
        backup_path = r'model/backup/' + self.model_name + '_' + strftime('%Y-%m-%d_%H-%M')
        with open(file_path, 'wb') as file:
            pickle.dump(trained_model, file)
        with open(backup_path, 'wb') as file:
            pickle.dump(trained_model, file)

    def load_pkl(self):
        file = open(self.filename, 'rb')
        tmp_dict = pickle.load(file)
        file.close()
        self.__dict__.update(tmp_dict)

    def save_pkl(self, file_path):
        file = open(file_path, 'wb')
        pickle.dump(self.__dict__, file, 2)
        file.close()