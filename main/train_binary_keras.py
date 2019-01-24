#!/usr/bin/python
# -*- coding: utf-8 -*-
import configparser
import pandas as pd
from lib.model.autoencoder import build_model
from lib.util.dataset import dataset, generate_test_users
from lib.util.preprocess import drop_features, extract_features_name, feature_engineering
from sklearn.metrics import f1_score, accuracy_score
from keras.models import load_model
import ast
import warnings
import argparse
import os
import numpy as np

warnings.filterwarnings("ignore")

DIR_FILE = '../data/config/dir_path.ini'
DIR_CONFIG = configparser.ConfigParser()
DIR_CONFIG.read(DIR_FILE)
DATA_PATH = DIR_CONFIG['DIR']['DATA_DIR']

TRAINING_EPOCHS = None
BATCH_SIZE = None
DISPLAY_STEP = None
TEST_USERS = None
TEST_DAYS = 30
RELOAD = False
MODEL_FILE = None
MODEL_CONFIG = None
TRAINING_DATA = None
TEST_USERS_PER_GROUP = None

def main():
    target_group_idx = 2

    parse_args()
    init_config()
    num_group = get_num_group()
    # search_useful_features(num_group)
    model, model_score = train_pipeline(num_group, target_group_idx)
    final_test(num_group, model, model_score, target_group_idx)

def final_test(num_group, model, model_score, target_group_idx):
    three_test_users_per_groups = generate_test_users(2)
    for target_group in range(num_group):
        target_group_dataset = dataset(test_users=three_test_users_per_groups, mode='test', days=TEST_DAYS,
                                       standardize=False, min_max=True)
        target_group_dataset.convert_to_unique_dataset(target_label=target_group, is_split=False)
        predict_group_id(model, model_score[0], target_group_dataset, target_group_idx)
        #metrics_validation(model, model_score[0], target_group_dataset, target_group_idx)

def predict_group_id(model, threshold, testing_data, target_group_idx):
    print('===== prediction stages =====')
    threshold = threshold * 1.5
    print("Threshold MSE: %.2f \n" % threshold)
    mse = validation(model, testing_data.naive_X)
    if mse < threshold:
        print('These user is belonging to group ' + str(target_group_idx) + ' ...')
    else:
        print('These user is belonging to other groups ...')

def metrics_validation(model, threshold, testing_data, target_group_idx):
    print('===== prediction stages =====')
    threshold = threshold * 1.5
    print("Threshold MSE: %.2f \n" % threshold)
    correct = 0
    for input_data, label in zip(testing_data.naive_X, testing_data.naive_y):
        input_data = np.array([input_data])
        mse = validation(model, input_data)
        if mse < threshold and label == target_group_idx:
            correct = correct + 1
        elif mse > threshold and label != target_group_idx:
            correct = correct + 1

    acc = correct/len(testing_data.naive_X)
    print('Accuracy for model_%d = %.2f' % (target_group_idx, acc))

def train_pipeline(num_group, target_group_idx):
    model_score = []
    for target_label in range(num_group):
        if target_label != target_group_idx:
            continue
        print('=============== model_' + str(target_label) + ' training ===============')
        training_data, validating_data, testing_data = init_dataset(num_group, target_label, load_data=False)
        model, test_mse = train_model(training_data, validating_data, testing_data)
        model_score.append(test_mse)
    return model, model_score

def search_useful_features(num_group):
    current_features = set(extract_features_name('template')[3:])
    drop_cols = set()
    drop_features(source_name='template', target_name='user_info', drop_feature=drop_cols)
    original_mse = train_pipeline(num_group)
    record_metrics = [original_mse[0]]
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
            test_mse = train_pipeline(num_group)
            temp_metrics.append(test_mse[0])
            temp_features.append(col)

        best_mse = min(temp_metrics)
        if best_mse < record_metrics[-1]:
            best_feature_idx = temp_metrics.index(best_mse)
            drop_cols.add(temp_features[best_feature_idx])
            record_metrics.append(best_mse)
            record_drop_features.append(drop_cols)
            current_features = current_features - drop_cols
        else:
            break
    print('=============== ' + 'redundant features' + ' ===============')
    print(record_drop_features)
    print('=============== ' + ' recorded metrics ' + ' ===============')
    print(record_metrics)

def get_num_group(file_name='user_group_relation'):
    file_path = DATA_PATH + file_name + '.csv'
    label_df = pd.read_csv(file_path)

    group_label = label_df.groupby('Group_ID')
    num_group = len(group_label.groups.keys())
    return num_group

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

    training_data.convert_to_unique_dataset(target_label, is_split=True)
    validating_data = {'input': training_data.X_test, 'label': training_data.y_test}
    testing_data.convert_to_unique_dataset(target_label, is_split=False)
    return training_data, validating_data, testing_data

def train_model(training_data, validating_data, testing_data):
    batch_x, batch_y = training_data.next_train_batch(1)
    input_shape = len(batch_x[0])
    model = init_model(input_shape, MODEL_CONFIG, reload=RELOAD)
    train(model, training_data, validating_data)
    print("========== Start testing ==========")
    test_mse = validation(model, testing_data.naive_X)
    return model, test_mse

def train(model, training_data, validating_data):
    print("\n ========== Start training ========== \n")
    for epoch in range(TRAINING_EPOCHS):
        total_batch = int(training_data.num_examples() / BATCH_SIZE)
        for i in range(total_batch):
            batch_x, __ = training_data.next_train_batch(BATCH_SIZE)
            model.train_on_batch(x=batch_x, y=batch_x)

        if epoch % DISPLAY_STEP == 0:
            print('===== Epoch: %04d validating =====' % (epoch + 1))
            validation(model, validating_data['input'])

    save_model(model, MODEL_CONFIG)
    print("Training phase finished... \n")

def validation(model, input_X):
    score = model.evaluate(input_X, input_X, verbose=1)
    mae = score[1]
    mse = score[2]
    print("Model validating MAE: %.2f" % mae)
    print("Model validating MSE: %.2f \n" % mse)
    return mse

def init_config():
    global TRAINING_EPOCHS, BATCH_SIZE, DISPLAY_STEP, TEST_USERS, TEST_DAYS, TEST_USERS_PER_GROUP
    TRAINING_EPOCHS = int(MODEL_CONFIG['train']['training_epochs'])
    BATCH_SIZE = int(MODEL_CONFIG['train']['batch_size'])
    DISPLAY_STEP = int(MODEL_CONFIG['train']['display_step'])
    TEST_USERS = ast.literal_eval(MODEL_CONFIG['train']['test_users'])
    TEST_USERS_PER_GROUP = int(MODEL_CONFIG['train']['test_users_per_group'])
    if len(TEST_USERS) < 1:
        TEST_USERS = generate_test_users(TEST_USERS_PER_GROUP)

    if MODEL_CONFIG['train']['test_users'] == 'None' or len(TEST_USERS) == 0:
        TEST_USERS = None
        TEST_DAYS = int(MODEL_CONFIG['train']['test_days'])

def parse_args():
    global MODEL_CONFIG, DATA_PATH, RELOAD, MODEL_FILE, TRAINING_DATA
    parser = argparse.ArgumentParser()
    required_arguments = parser.add_argument_group('require_arguments')
    optional_arguments = parser.add_argument_group('optional_arguments')

    required_arguments.add_argument("-m", "--model-config", help="model config file", required=True)

    optional_arguments.add_argument("-r", "--reload-model", help="reload model", action='store_true')
    optional_arguments.add_argument("-d", "--reload-training_data", help="reload training data")

    args = parser.parse_args()
    RELOAD = args.reload_model
    TRAINING_DATA = args.reload_training_data
    MODEL_FILE = '../data/config/' + args.model_config + '.ini'
    if not os.path.isfile(MODEL_FILE):
        raise ValueError('File not exist: ' + MODEL_FILE)
    MODEL_CONFIG = configparser.ConfigParser()
    MODEL_CONFIG.read(MODEL_FILE)

def save_model(model, config):
    path = DATA_PATH + 'result/model/' + config['model']['name'] + '.h5'
    model.save(path)
    print("=== Model saved ===")

def init_model(input_shape, config, reload=False):
    if reload:
        path = DATA_PATH + 'result/model/' + config['model']['name'] + '.h5'
        model = load_model(path)
        print("=== Model restored ===")
    else:
        model = build_model(input_shape)
    print (model.summary())
    return model

if __name__ == '__main__':
    main()