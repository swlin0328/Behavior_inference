#!/usr/bin/python
# -*- coding: utf-8 -*-
import configparser
import tensorflow as tf
from lib.model.branch_DNN import build_model
from lib.util.dataset import dataset, generate_test_users
from lib.util.preprocess import drop_features, extract_features_name, feature_engineering
from sklearn.metrics import f1_score, accuracy_score
import ast
import warnings
import argparse
import os
import pandas as pd

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
    parse_args()
    init_config()
    search_useful_features()

def train_pipeline():
    training_data, validating_data, testing_data = init_dataset()
    validate_f1, test_f1 = train_model(training_data, validating_data, testing_data)
    return test_f1

"""
def create_useful_features():
    current_features = extract_features_name('template')[3:28]
    source_path = DATA_PATH + 'template.csv'
    target_path = DATA_PATH + 'user_info.csv'
    df = pd.read_csv(source_path)

    useful_attr = []
    original_acc = train_pipeline()
    record_metrics = [original_acc]
    record_features_creation = []
    operator = ['*']

    for idx, attr_1 in enumerate(current_features):
        for attr_2 in current_features[idx:]:
            for op in operator:
                tf.reset_default_graph()

                for creator in record_features_creation:
                    df, _ = feature_engineering(df, attribute_1=creator[0], attribute_2=creator[1], operator=creator[2])

                feature_creator = [attr_1, attr_2, op]
                df, new_feature = feature_engineering(df, attribute_1=feature_creator[0],
                                                      attribute_2=feature_creator[1], operator=feature_creator[2])
                df.to_csv(target_path, encoding='utf_8_sig', index=False)
                test_acc = train_pipeline()
                if test_acc > record_metrics[-1]:
                    record_metrics.append(test_acc)
                    useful_attr.append(new_feature)
                    record_features_creation.append(feature_creator)

    print('=============== ' + 'created features' + ' ===============')
    print(useful_attr)
    print('=============== ' + ' recorded metrics ' + ' ===============')
    print(record_metrics)
"""

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
            tf.reset_default_graph()
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
    if TRAINING_DATA is None:
        print("\n preparing training dataset...")
        training_data = dataset(test_users=TEST_USERS, days=TEST_DAYS, standardize=False, min_max=True)
        training_data.split_dataset(concat=False, one_hot=True)
        training_data.save_pkl('training_data')
    else:
        training_data = dataset(pkl_file=TRAINING_DATA)

    validating_X, validating_y = training_data.get_non_concate_dataset(training_data.X_test, training_data.y_test)
    validating_data = {'input': validating_X, 'label': validating_y}
    one_hot_enc = training_data.get_dataprocessor()

    print("\n preparing testing dataset...")
    testing_data = dataset(test_users=TEST_USERS, mode='test', enc=one_hot_enc, days=TEST_DAYS, standardize=False, min_max=True)
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