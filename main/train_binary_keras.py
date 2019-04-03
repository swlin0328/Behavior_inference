#!/usr/bin/python
# -*- coding: utf-8 -*-
from ..main.lib.util.dataset import dataset
from ..main.lib.util.preprocess import get_num_group, generate_test_users
from ..main.lib.util.validation import mse_evaluate, anomaly_metrics_validation
from ..main.lib.config.config import DATA_PATH, set_tf_session
from ..main.lib.db.dnn_model import sql4Keras
from ..main.lib.db.dataset import sql4Dataset
from keras.models import load_model
import numpy as np
import copy
import math
import os


def start(config, db_conn):
    num_group = get_num_group()
    naive_dataset = init_dataset(config, db_conn)
    train_pipeline(naive_dataset, num_group, config, db_conn)


def train_pipeline(naive_dataset, num_groups, config, db_conn=None):
    for group_idx in range(num_groups):
        print('=============== model_' + str(group_idx) + ' training ===============')
        training_dataset, validating_dataset, testing_dataset = convert_unique_dataset(group_idx, naive_dataset)
        model, num_step = train_model(training_dataset, validating_dataset, config)

        print("========== Start testing ==========")
        test_mse = mse_evaluate(model, testing_dataset.naive_X)
        acc, f1_score = anomaly_metrics_validation(model, test_mse, testing_dataset, group_idx)
        metrics = [test_mse, acc]

        sql4model = sql4Keras(model_name=config['model_name'], customer_group=group_idx, creator=config['user'],
                              sql_conn=db_conn)
        sql4model.save2sql(model, model_type=config['model_type'], valid_metrics=metrics, step=num_step)


def init_dataset(config, db_conn):
    test_users = generate_test_users(config['test_users_per_group'])
    print("\n preparing training dataset...")
    training_data = dataset(config, test_users=test_users, standardize=False, min_max=True)
    print("\n preparing testing dataset...")
    testing_data = dataset(config, test_users=test_users, mode='test', standardize=False, min_max=True)

    naive_dataset = pca_naive_dataset(training_data, testing_data, config, db_conn)
    return naive_dataset


def pca_naive_dataset(training_data, testing_data, config, db_conn):
    training_data.pca_naive_X(pca_dim=25)
    pca = training_data.get_pca_tranducer()

    testing_data.pca_naive_X(pca_transducer=pca)
    sql_dataset = sql4Dataset(dataset_name=config['model_name'], sql_conn=db_conn)
    sql_dataset.save2sql(pca)
    naive_dataset = {'train': training_data, 'test': testing_data}
    return naive_dataset


def convert_unique_dataset(target_label, naive_dataset):
    temp_dataset = copy.deepcopy(naive_dataset)
    temp_dataset['train'].convert_to_unique_label_dataset(target_label, is_split=True)
    temp_dataset['test'].convert_to_unique_label_dataset(target_label, is_split=False)

    training_data = temp_dataset['train']
    validating_data = {'input': temp_dataset['train'].X_test, 'label': temp_dataset['train'].y_test}
    testing_data = temp_dataset['test']
    return training_data, validating_data, testing_data


def train_model(training_dataset, validating_data, config):
    batch_x, batch_y = training_dataset.next_train_batch(1)
    input_shape = len(batch_x[0])
    model = init_model(config, input_shape, config['model_name'])

    batch_size = int(config['batch_size'])
    training_epochs = int(config['training_epochs'])
    num_batch = math.floor(training_dataset.num_examples() / batch_size)
    num_step = num_batch * training_epochs
    print("\n ========== Start training ========== \n")
    for epoch in range(training_epochs):
        for i in range(num_batch):
            batch_x, __ = training_dataset.next_train_batch(batch_size)
            model.train_on_batch(x=batch_x, y=batch_x)

        if epoch % int(config['display_step']) == 0:
            print('===== Epoch: %04d validating =====' % (epoch + 1))
            mse_evaluate(model, validating_data['input'])
    save_model(model, config['model_name'])
    print("Training phase finished... \n")
    return model, num_step


def init_model(config, input_shape, model_name):
    set_tf_session(config)

    if config['reload_model'] == 'True':
        path = DATA_PATH + 'result/model/' + model_name + '.h5'
        model = load_model(path)
        print("=== Model restored ===")
    else:
        model = build_model(config, input_shape, encoding_dim=6)
    print(model.summary())
    return model


def save_model(model, model_name):
    path = DATA_PATH + 'result/model/' + model_name + '.h5'
    model.save(path)
    print("=== Model saved ===")


def build_model(config, input_shape, encoding_dim):
    path = r'./inference/main/lib/model/'
    dir_path = path + config['model_name']
    if not os.path.isdir(dir_path):
        from ..main.lib.model.keras_autoencoder import build_model
        return build_model(input_shape, encoding_dim)
    else:
        import sys
        sys.path.append(dir_path)
        from customized_model import build_model
        return build_model(input_shape, encoding_dim)


if __name__ == '__main__':
    start()