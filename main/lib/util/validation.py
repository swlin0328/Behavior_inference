#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import configparser
import datetime
from ..config.config import DATA_PATH
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf


class energy_metrics:
    def __init__(self, cluster_file='group_center.csv', load_file='for_clustering.csv'):
        self.group_center = pd.read_csv(DATA_PATH + cluster_file)
        self.household_consumption = pd.read_csv(DATA_PATH + load_file)
        self.kwh_factor = 15 * 60 / (1000 * 3600)

    def mean_absolute_error(self, prediction, test_users):
        absolute_error = 0
        num_estimation = prediction.shape[0]

        user_consumption = self.household_consumption.groupby('User_ID')
        group_consumption = self.group_center.sum(axis=1) / self.kwh_factor
        real_consumption = []
        for user in test_users:
            household = user_consumption.get_group(user)
            mean_consumption = household.iloc[:, 3:99].sum(axis=1).mean()
            real_consumption.append(mean_consumption)

        real_consumption = np.array(real_consumption).mean() / self.kwh_factor
        for label in prediction:
            predict_consumption = group_consumption[label]
            absolute_error = absolute_error + abs(predict_consumption-real_consumption)

        mae = absolute_error/num_estimation
        return mae


def mse_evaluate(model, input_X, verbose=1):
    score = model.evaluate(input_X, input_X, verbose=verbose)
    mae = score[1]
    mse = score[2]
    if verbose:
        print("Model validating MAE: %.2f" % mae)
        print("Model validating MSE: %.2f \n" % mse)
    return mse


def anomaly_metrics_validation(model, threshold, testing_data, target_group_idx):
    print('===== prediction stages =====')
    threshold = threshold * 1.2
    predict_y = predict_binary_label(model, threshold, testing_data, target_group_idx)
    validate_acc = accuracy_score(testing_data.naive_y, predict_y)
    validate_f1 = f1_score(testing_data.naive_y, predict_y, labels=[target_group_idx], average='macro')
    print('Accuracy for model_%d = %.2f' % (target_group_idx, validate_acc))
    return validate_acc, validate_f1


def predict_binary_label(model, threshold, testing_data, target_group_idx):
    results = []
    for input_data, label in zip(testing_data.naive_X, testing_data.naive_y):
        input_data = np.array([input_data])
        mse = mse_evaluate(model, input_data, verbose=0)
        if mse < threshold:
            results.append(target_group_idx)
        else:
            results.append(-1)
    return results


def tf_validation(sess, model_config, input_user, input_tax, input_weather, label_y):
    predict_y = tf.cast(tf.argmax(model_config['output'], 1), tf.int32, name='predictions')
    true_y = tf.cast(tf.argmax(label_y, 1), tf.int32, name='true_labels')
    predict_y, true_y = sess.run([predict_y, true_y],
                                 feed_dict={model_config['input']['user']: input_user,
                                            model_config['input']['tax']: input_tax,
                                            model_config['input']['weather']: input_weather,
                                            model_config['input']['group_label']: label_y,
                                            model_config['keep_prob']: 1.0,
                                            model_config['batch_size']: len(input_user)})
    validate_acc = accuracy_score(true_y, predict_y)
    validate_f1 = f1_score(true_y, predict_y, average='macro')
    print("Model validating Accuracy: %.2f \n" % validate_acc)
    print("Model validating F1_score: %.2f \n" % validate_f1)
    return validate_acc, validate_f1
