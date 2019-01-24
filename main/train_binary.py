#!/usr/bin/python
# -*- coding: utf-8 -*-
import configparser
import tensorflow as tf
import pandas as pd
from lib.model.branch_DNN import build_model
from lib.util.dataset import dataset
from lib.util.dataset import generate_test_users
from lib.util.preprocess import drop_features, extract_features_name, feature_engineering
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
import ast
import warnings
import argparse
import os

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
    target_group_idx = 1

    parse_args()
    init_config()
    num_group = get_num_group()
    enc = init_binary_encoder()
    search_useful_features(target_group_idx, num_group, enc)

def train_pipeline(target_group_idx, num_group, enc):
    model_score = []
    for target_label in range(num_group):
        if target_label != target_group_idx:
            continue
        print('=============== model_' + str(target_label) + ' training ===============')
        tf.reset_default_graph()
        training_data, validating_data, testing_data = init_dataset(num_group, target_label, enc, load_data=False)
        validate_f1, test_f1 = train_model(training_data, validating_data, testing_data)
        model_score.append(test_f1)
    return model_score

def search_useful_features(target_group_idx, num_group, enc):
    current_features = set(extract_features_name('template')[3:])
    drop_cols = set()
    drop_features(source_name='template', target_name='user_info', drop_feature=drop_cols)
    original_acc = train_pipeline(target_group_idx, num_group, enc)
    record_metrics = [original_acc[0]]
    record_drop_features = ['None']
    while True:
        temp_metrics = []
        temp_features = []
        for col in current_features:
            tf.reset_default_graph()
            temp_drop = drop_cols.copy()
            temp_drop.add(col)
            print('=============== current dropped columns ===============')
            print(temp_drop)

            drop_features(source_name='template', target_name='user_info', drop_feature=temp_drop)
            test_acc = train_pipeline(target_group_idx, num_group, enc)
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

def get_num_group(file_name='user_group_relation'):
    file_path = DATA_PATH + file_name + '.csv'
    label_df = pd.read_csv(file_path)

    group_label = label_df.groupby('Group_ID')
    num_group = len(group_label.groups.keys())
    return num_group

def init_binary_encoder():
    enc = OneHotEncoder()
    enc.fit([[0], [1]])
    return enc

def init_dataset(num_group, target_label, enc, load_data):
    train_file_name = "original_training_data_" + str(num_group)
    test_file_name = "original_testing_data_" + str(num_group)
    pkl_path = DATA_PATH + 'result/dataset/'
    if not os.path.isfile(pkl_path + train_file_name + '.pkl') or not os.path.isfile(
            pkl_path + test_file_name + '.pkl') or not load_data:
        print("\n preparing training dataset...")
        training_data = dataset(test_users=TEST_USERS, days=TEST_DAYS, standardize=False, min_max=True, enc=enc)
        training_data.save_pkl(train_file_name)
        print("\n preparing testing dataset...")
        testing_data = dataset(test_users=TEST_USERS, mode='test', enc=enc, days=TEST_DAYS, standardize=False, min_max=True)
        testing_data.save_pkl(test_file_name)
    else:
        training_data = dataset(pkl_file=train_file_name)
        testing_data = dataset(pkl_file=test_file_name)

    training_data.convert_to_binary_label(target_label)
    training_data.split_dataset(concat=False, one_hot=True)

    validating_X, validating_y = training_data.get_non_concate_dataset(training_data.X_test, training_data.y_test)
    validating_data = {'input': validating_X, 'label': validating_y}

    testing_data.convert_to_binary_label(target_label)
    testing_data.generate_naive_dataset(concat=False, one_hot=True)
    return training_data, validating_data, testing_data

def train_model(training_data, validating_data, testing_data):
    config = build_model(training_data, MODEL_FILE)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_model(sess, saver, config, reload=RELOAD)
        train(sess, config, training_data, saver)
        print("========== Start validating ==========")
        validate_acc, validate_f1 = validation(sess, config,
                                               validating_data['input']['BI'],
                                               validating_data['input']['tax'],
                                               validating_data['input']['weather'],
                                               validating_data['label'])

        print("========== Start testing ==========")
        test_acc, test_f1 = validation(sess, config,
                                       testing_data.naive_X['BI'],
                                       testing_data.naive_X['tax'],
                                       testing_data.naive_X['weather'],
                                       testing_data.naive_y)
        return validate_f1, test_f1

def train(sess, config, training_data, saver):
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/TensorBoard', sess.graph)
    print("\n ========== Start training ========== \n")
    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0
        total_batch = int(training_data.num_examples() / BATCH_SIZE)
        for i in range(total_batch):
            batch_x, batch_ys = training_data.next_train_batch(BATCH_SIZE)
            ___, summary = sess.run([config['optimizer'], merged], feed_dict={config['input']['BI']: batch_x['BI'],
                                                                    config['input']['tax']: batch_x['tax'],
                                                                    config['input']['weather']: batch_x['weather'],
                                                                    config['input']['group_label']: batch_ys,
                                                                    config['keep_prob']: 0.8,
                                                                    config['batch_size']: BATCH_SIZE})
            avg_cost += sess.run(config['cost'], feed_dict={config['input']['BI']: batch_x['BI'],
                                                  config['input']['tax']: batch_x['tax'],
                                                  config['input']['weather']: batch_x['weather'],
                                                  config['input']['group_label']: batch_ys,
                                                    config['keep_prob']: 0.8,
                                                    config['batch_size']: BATCH_SIZE})
        if epoch % DISPLAY_STEP == 0:
            train_writer.add_summary(summary, epoch)
            print('Epoch: %04d cost= %.9f' % ((epoch + 1), avg_cost))

    train_writer.close()
    save_model(sess, saver, model_name=config['model_name'])
    print("Training phase finished... \n")

def validation(sess, config, input_BI, input_tax, input_weather, label_y):
    predict_y = tf.cast(tf.argmax(config['output'], 1), tf.int32, name='predictions')
    true_y = tf.cast(tf.argmax(label_y, 1), tf.int32, name='true_labels')
    #correct_prediction = tf.equal(predict_y, true_y)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    predict_y, true_y = sess.run([predict_y, true_y],
                                 feed_dict={config['input']['BI']: input_BI,
                                            config['input']['tax']: input_tax,
                                            config['input']['weather']: input_weather,
                                            config['input']['group_label']: label_y,
                                            config['keep_prob']: 1.0,
                                            config['batch_size']: len(input_BI)})
    validate_acc = accuracy_score(true_y, predict_y)
    validate_f1 = f1_score(true_y, predict_y, average='macro')
    print("Model validating Accuracy: %.2f \n" % validate_acc)
    print("Model validating F1_score: %.2f \n" % validate_f1)
    return validate_acc, validate_f1

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

def save_model(sess, saver, model_name='test_model'):
    path = DATA_PATH + 'result/model/' + model_name + '.ckpt'
    saver.save(sess, path)
    print("=== Model saved ===")

def init_model(sess, saver, config, reload=False):
    sess.run(config['init'])
    if reload:
        path = DATA_PATH + 'result/model/' + config['model_name'] + '.ckpt'
        saver.restore(sess, path)
        print("=== Model restored ===")

if __name__ == '__main__':
    main()