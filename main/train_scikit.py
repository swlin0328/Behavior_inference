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
warnings.filterwarnings("ignore")

MODEL_FILE = '../data/config/model_config.ini'
MODEL_CONFIG = configparser.ConfigParser()
MODEL_CONFIG.read(MODEL_FILE)

TEST_USERS = None
TEST_DAYS = None
DATASET = None
TEST_USERS_PER_GROUP = None

def main():
    init_config()
    search_useful_features()


def train_pipeline():
    training_data, testing_data = init_dataset()
    model = train_model(training_data)
    predict_y = model.predict(testing_data.naive_X)
    test_acc = accuracy_score(testing_data.naive_y, predict_y)
    test_f1 = f1_score(testing_data.naive_y, predict_y, average='macro')
    print('Accuracy for testing data: %.3f' % test_acc)
    print('F1_score for testing data: %.3f' % test_f1)
    return test_f1


def search_useful_features():
    current_features = set(extract_features_name('template')[3:])
    drop_cols = set()
    drop_features(source_name='template', target_name='user_info', drop_feature=drop_cols)
    original_acc = train_pipeline()
    record_metrics = [original_acc]
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
            test_acc = train_pipeline()
            temp_metrics.append(test_acc)
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


def init_dataset():
    print("\npreparing training dataset...")
    training_data = dataset(test_users=TEST_USERS, days=TEST_DAYS, standardize=False, min_max=True)
    training_data.split_dataset(is_concat=True, is_one_hot=False)

    print("\npreparing testing dataset...")
    testing_data = dataset(test_users=TEST_USERS, mode='test', days=TEST_DAYS, standardize=False, min_max=True)
    testing_data.generate_naive_dataset(is_concat=True, is_one_hot=False)
    return training_data, testing_data

def train_model(training_data):
    #pipe_lr = Pipeline([('pca', PCA(n_components=30)), ('clf', SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0))])
    pipe_lr = Pipeline([('clf', SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0))])

    print('\n========== Starting training ==========\n')
    pipe_lr.fit(training_data.X_train, training_data.y_train)

    print('Accuracy for training data: %.3f' % pipe_lr.score(training_data.X_test, training_data.y_test))
    scores = cross_val_score(estimator=pipe_lr, X=training_data.X_train, y=training_data.y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    print('\n========== Training stage is complete ==========\n')
    return pipe_lr

def init_config():
    global TEST_USERS, TEST_DAYS, TEST_USERS_PER_GROUP
    TEST_USERS = ast.literal_eval(MODEL_CONFIG['train']['test_users'])
    TEST_USERS_PER_GROUP = int(MODEL_CONFIG['train']['test_users_per_group'])
    if len(TEST_USERS) < 1:
        TEST_USERS = generate_test_users(TEST_USERS_PER_GROUP)

    if MODEL_CONFIG['train']['test_users'] == 'None' or len(TEST_USERS) == 0:
        TEST_USERS = None
        TEST_DAYS = int(MODEL_CONFIG['train']['test_days'])

if __name__ == '__main__':
    main()