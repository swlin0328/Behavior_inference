#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from ..main.lib.util.dataset import dataset
from ..main.lib.util.preprocess import generate_test_users
from ..main.lib.db.svm_model import sql4SVM
from ..main.lib.db.dataset import sql4Dataset
import warnings


def start(config, db_conn):
    warnings.filterwarnings("ignore")
    training_data, testing_data = init_dataset(config, db_conn)
    train_pipeline(training_data, testing_data, config, db_conn)


def init_dataset(config, db_conn):
    test_users = generate_test_users(config['test_users_per_group'])
    print("\n preparing training dataset...")
    training_data = dataset(config, test_users=test_users, standardize=False, min_max=True)
    print("\n preparing testing dataset...")
    testing_data = dataset(config, test_users=test_users, mode='test', standardize=False, min_max=True)

    training_data, testing_data = pca_naive_dataset(training_data, testing_data, config, db_conn)
    return training_data, testing_data


def pca_naive_dataset(training_data, testing_data, config, db_conn):
    training_data.pca_naive_X(pca_dim=25)
    pca = training_data.get_pca_tranducer()

    testing_data.pca_naive_X(pca_transducer=pca)
    training_data.split_dataset(test_size=0.1)
    sql_dataset = sql4Dataset(dataset_name=config['model_name'], sql_conn=db_conn)
    sql_dataset.save2sql(pca)
    return training_data, testing_data


def train_pipeline(training_data, testing_data, config, db_conn):
    model = train_model(training_data)
    predict_y = model.predict(testing_data.naive_X)
    test_acc = accuracy_score(testing_data.naive_y, predict_y)
    print('Accuracy for testing data: %.3f' % test_acc)

    sql4model = sql4SVM(model_name=config['model_name'], customer_group=-1, creator=config['user'], sql_conn=db_conn)
    sql4model.save2sql(model, model_type=config['model_type'], valid_metrics=test_acc)


def train_model(training_data):
    svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)

    print('\n========== Starting training ==========\n')
    svm.fit(training_data.X_train, training_data.y_train)

    print('Accuracy for training data: %.3f' % svm.score(training_data.X_test, training_data.y_test))
    scores = cross_val_score(estimator=svm, X=training_data.X_train, y=training_data.y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    print('\n========== Training stage is complete ==========\n')
    return svm


if __name__ == '__main__':
    start()
