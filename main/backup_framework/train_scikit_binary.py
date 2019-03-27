#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from lib.util.preprocess import drop_features, extract_features_name, feature_engineering
import configparser
from lib.util.dataset import dataset, generate_test_users
import ast
import warnings
import pandas as pd
import os

warnings.filterwarnings("ignore")

MODEL_FILE = '../data/config/model_config.ini'
MODEL_CONFIG = configparser.ConfigParser()
MODEL_CONFIG.read(MODEL_FILE)

DIR_FILE = '../data/config/dir_path.ini'
DIR_CONFIG = configparser.ConfigParser()
DIR_CONFIG.read(DIR_FILE)

TEST_USERS = None
TEST_DAYS = None
DATASET = None
TEST_USERS_PER_GROUP = None
DATA_PATH = None

def main():
    target_group_idx = 0

    init_config()
    num_group = get_num_group()
    train_pipeline(target_group_idx, num_group)
    #search_useful_features(target_group_idx, num_group)


def train_pipeline(target_group_idx, num_group):
    model_score = []
    for target_label in range(num_group):
        if target_label != target_group_idx:
            continue
        print('=============== model_' + str(target_label) + ' training ===============')
        training_data, testing_data = init_dataset(num_group, target_label, load_data=False)
        model = train_model(training_data)

        predict_y = model.predict(testing_data.naive_X)
        test_acc = accuracy_score(testing_data.naive_y, predict_y)
        test_f1 = f1_score(testing_data.naive_y, predict_y)
        model_score.append(test_f1)
        print('Accuracy for testing data: %.3f' % test_acc)
        print('F1_score for testing data: %.3f' % test_f1)
    return model_score


def search_useful_features(target_group_idx, num_group):
    current_features = set(extract_features_name('template')[3:])
    drop_cols = set()
    drop_features(source_name='template', target_name='user_info', drop_feature=drop_cols)
    original_acc = train_pipeline(target_group_idx, num_group)
    record_metrics = [original_acc[0]]
    record_drop_features = ['None']
    while True:
        temp_metrics = []
        temp_features = []
        for col in current_features:
            temp_drop = drop_cols.copy()
            temp_drop.add(col)
            print('=============== current dropped columns ===============')
            print(temp_drop)

            drop_features(source_name='template', target_name='user_info', drop_feature=temp_drop)
            test_acc = train_pipeline(target_group_idx, num_group)
            temp_metrics.append(test_acc[0])
            temp_features.append(col)

        best_acc = max(temp_metrics)
        if best_acc > record_metrics[-1]:
            best_feature_idx = temp_metrics.index(best_acc)
            drop_cols.add(temp_features[best_feature_idx])
            record_metrics.append(best_acc)
            record_drop_features.append(drop_cols)
            current_features = current_features - drop_cols
        else:
            break
    print('=============== ' + 'redundant features' + ' ===============')
    print(record_drop_features)
    print('=============== ' + ' recorded metrics ' + ' ===============')
    print(record_metrics)


def init_dataset(num_group, target_label, load_data):
    train_file_name = "original_training_data_" + str(num_group)
    test_file_name = "original_testing_data_" + str(num_group)
    pkl_path = DATA_PATH + 'result/dataset/'
    if not os.path.isfile(pkl_path + train_file_name + '.pkl') or not os.path.isfile(
            pkl_path + test_file_name + '.pkl') or not load_data:
        print("\n preparing training dataset...")
        training_data = dataset(test_users=TEST_USERS, days=TEST_DAYS, standardize=False, min_max=True)
        training_data.save_pkl(train_file_name)
        print("\n preparing testing dataset...")
        testing_data = dataset(test_users=TEST_USERS, mode='test', days=TEST_DAYS, standardize=False, min_max=True)
        testing_data.save_pkl(test_file_name)
    else:
        training_data = dataset(pkl_file=train_file_name)
        testing_data = dataset(pkl_file=test_file_name)

    training_data.convert_to_binary_label(target_label)
    training_data.split_dataset(is_concat=True, is_one_hot=False)

    testing_data.convert_to_binary_label(target_label)
    testing_data.generate_naive_dataset(is_concat=True, is_one_hot=False)
    return training_data, testing_data


def get_num_group(file_name='user_group_relation'):
    file_path = DATA_PATH + file_name + '.csv'
    label_df = pd.read_csv(file_path)

    group_label = label_df.groupby('Group_ID')
    num_group = len(group_label.groups.keys())
    return num_group


def train_model(training_data):
    pipe_lr = Pipeline([('clf', SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0))])

    print('\n========== Starting training ==========\n')
    pipe_lr.fit(training_data.X_train, training_data.y_train)

    print('Accuracy for training data: %.3f' % pipe_lr.score(training_data.X_test, training_data.y_test))
    scores = cross_val_score(estimator=pipe_lr, X=training_data.X_train, y=training_data.y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    print('\n========== Training stage is complete ==========\n')
    return pipe_lr


def init_config():
    global TEST_USERS, TEST_DAYS, TEST_USERS_PER_GROUP, DATA_PATH
    TEST_USERS = ast.literal_eval(MODEL_CONFIG['train']['test_users'])
    TEST_USERS_PER_GROUP = int(MODEL_CONFIG['train']['test_users_per_group'])
    DATA_PATH = DIR_CONFIG['DIR']['DATA_DIR']
    if len(TEST_USERS) < 1:
        TEST_USERS = generate_test_users(TEST_USERS_PER_GROUP)

    if MODEL_CONFIG['train']['test_users'] == 'None' or len(TEST_USERS) == 0:
        TEST_USERS = None
        TEST_DAYS = int(MODEL_CONFIG['train']['test_days'])


if __name__ == '__main__':
    main()