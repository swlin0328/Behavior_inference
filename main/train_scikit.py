#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
import configparser
from lib.util.dataset import dataset
import ast
import warnings
warnings.filterwarnings("ignore")

MODEL_FILE = '../data/config/tf_model_config.ini'
MODEL_CONFIG = configparser.ConfigParser()
MODEL_CONFIG.read(MODEL_FILE)
TEST_USERS = None
TEST_DAYS = 30

def main():
    init_config()
    training_data, testing_data = init_dataset()
    model = train_model(training_data)
    print('Accuracy for testing data: %.3f' % model.score(testing_data.naive_X, testing_data.naive_y))

def init_dataset():
    print("\npreparing training dataset...")
    training_data = dataset(test_users=TEST_USERS, days=TEST_DAYS)
    training_data.split_dataset(concat=True, one_hot=False)

    print("\npreparing testing dataset...")
    testing_data = dataset(test_users=TEST_USERS, mode='test', days=TEST_DAYS)
    testing_data.generate_naive_dataset(concat=True, one_hot=False)
    return training_data, testing_data

def train_model(training_data):
    pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=10)),
                        ('clf', SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0))])

    print('\n========== Starting training ==========\n')
    pipe_lr.fit(training_data.X_train, training_data.y_train)

    print('Accuracy for training data: %.3f' % pipe_lr.score(training_data.X_test, training_data.y_test))
    scores = cross_val_score(estimator=pipe_lr, X=training_data.X_train, y=training_data.y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    print('\n========== Training stage is complete ==========\n')
    return pipe_lr

def init_config():
    global TEST_USERS, TEST_DAYS
    TEST_USERS = ast.literal_eval(MODEL_CONFIG['train']['test_users'])
    if MODEL_CONFIG['train']['test_users'] == 'None' or len(TEST_USERS) == 0:
        TEST_USERS = None
        TEST_DAYS = int(MODEL_CONFIG['train']['test_days'])

if __name__ == '__main__':
    main()