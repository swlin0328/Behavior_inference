#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import pickle
from ..util.source import data_extractor
from sklearn.decomposition import PCA
import configparser
from ..config.config import DATA_PATH

class dataset:
    def __init__(self, config, pkl_file=None, cluster_file='user_group_relation.csv', user_file='user_info.csv',
                 tax_file='city_tax.csv', weather_file='city_weather.csv', test_users=None, days=30,
                 mode='train', enc=None, standardize=False, min_max=False):
        self.config = config
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
        self.pca = None
        self.indices = None

        self.init_datasource(self.config,
                             cluster_file, user_file, tax_file, weather_file,
                             test_users, days, mode, standardize, min_max)

    def init_datasource(self, config,
                        cluster_file, user_file, tax_file, weather_file, test_users, days, mode, standardize, min_max):
        data_source = data_extractor(config, cluster_file, user_file, tax_file, weather_file, mode=mode)
        data = data_source.init_with_csv(test_users, days, standardize, min_max)
        self.input['user'] = data['user']
        self.input['tax'] = data['tax']
        self.input['weather'] = data['weather']
        self.label = data['group_label']
        self.user_size = data['user_size']
        self.tax_size = data['tax_size']
        self.weather_size = data['weather_size']
        self.data_size = len(self.label)

        if len(self.input['user']) != self.data_size or len(
                self.input['tax']) != self.data_size or len(self.input['weather']) != self.data_size:
            raise ValueError('Illegal dataset size!')

    def num_examples(self):
        return self.data_size

    def get_dataprocessor(self):
        return self.enc

    def get_pca_tranducer(self):
        return self.pca

    def concat_input_data(self):
        if len(self.concat_input) == 0:
            for idx in range(len(self.input['user'])):
                concat_data = np.concatenate([self.input['user'][idx], self.input['tax'][idx],
                                              self.input['weather'][idx]], axis=0)
                self.concat_input.append(concat_data)
        return self.concat_input

    def generate_naive_dataset(self, is_concat=True, is_one_hot=False):
        if self.data_size == 0:
            raise ValueError('Empty dataset!')
        self.generate_naive_label(is_one_hot)
        self.generate_naive_X(is_concat)

    def generate_naive_X(self, is_concat):
        if is_concat:
            self.naive_X = self.concat_input_data()
            self.is_concat = True
            return

        self.naive_X = []
        if self.mode == 'test':
            self.naive_X = self.input
            return
        else:
            for idx in range(len(self.input['user'])):
                self.naive_X.append([self.input['user'][idx], self.input['tax'][idx], self.input['weather'][idx]])

    def generate_naive_label(self, is_one_hot):
        if is_one_hot:
            if self.enc == None and self.mode == 'train':
                self.enc = OneHotEncoder(handle_unknown='ignore')
                self.naive_y = self.enc.fit_transform(np.array(self.label).reshape(-1, 1)).toarray()
            else:
                self.naive_y = self.enc.transform(np.array(self.label).reshape(-1, 1)).toarray()
        else:
            self.naive_y = self.label

    def split_dataset(self, test_size=0.1):
        if self.is_concat == False:
            raise ValueError('Dataset is not concat format!')

        if len(self.input['user']) != self.data_size or len(
                self.input['tax']) != self.data_size or len(self.input['weather']) != self.data_size:
            raise ValueError('Illegal dataset size!')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.naive_X, self.naive_y,
                                                                                test_size=test_size, random_state=0)

    def next_train_batch(self, batch_size):
        self.batch_idx = self.batch_idx + batch_size
        if self.y_train is None or self.X_train is None:
            print('Empty training dataset!')
            return
        if self.indices is None:
            self.init_batch_idx()
        if self.batch_idx > len(self.X_train):
            self.reset_batch_idx(batch_size)

        batch_x = self.X_train[self.batch_idx - batch_size:self.batch_idx]
        batch_y = self.y_train[self.batch_idx - batch_size:self.batch_idx]
        if self.is_concat:
            return batch_x, batch_y
        else:
            return self.get_non_concate_dataset(batch_x, batch_y)

    def reset_batch_idx(self, batch_size):
        self.batch_idx = batch_size
        np.random.shuffle(self.indices)

    def init_batch_idx(self):
        self.indices = np.arange(len(self.X_train))
        np.random.shuffle(self.indices)

    def get_non_concate_dataset(self, input_x, label_y):
        datasize = len(input_x)
        if len(label_y) != datasize:
            raise ValueError('Invalid dataset!')
        x_user = []
        x_tax = []
        x_weather = []
        for idx in range(datasize):
            x_user.append(input_x[idx][0])
            x_tax.append(input_x[idx][1])
            x_weather.append(input_x[idx][2])
        return {'user': x_user, 'tax': x_tax, 'weather': x_weather}, label_y

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

    def convert_to_unique_label_dataset(self, target_label, is_split=False, test_size=0.1):
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

    def pca_naive_X(self, is_concat=True, is_one_hot=False, pca_transducer=None, pca_dim=50):
        self.generate_naive_dataset(is_concat, is_one_hot)
        self.pca = pca_transducer
        if self.pca is None:
            self.pca = PCA(n_components=pca_dim)
            self.naive_X = self.pca.fit_transform(self.naive_X)
        else:
            self.naive_X = self.pca.transform(self.naive_X)

