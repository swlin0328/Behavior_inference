#!/usr/bin/python
# -*- coding: utf-8 -*-
import configparser
import tensorflow as tf
from lib.model.branch_DNN import build_model
from lib.util.dataset import dataset
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

def main():
    parse_args()
    init_config()
    training_data, validating_data, testing_data = init_dataset()
    train_model(training_data, validating_data, testing_data)

def init_dataset():
    print("\n preparing training dataset...")
    training_data = dataset(test_users=TEST_USERS, days=TEST_DAYS, standardize=False, Min_Max=True)
    training_data.split_dataset(concat=False, one_hot=True)

    validating_X, validating_y = training_data.get_non_concate_dataset(training_data.X_test, training_data.y_test)
    validating_data = {'input': validating_X, 'label': validating_y}
    one_hot_enc = training_data.get_dataprocessor()

    print("\n preparing testing dataset...")
    testing_data = dataset(test_users=TEST_USERS, mode='test', enc=one_hot_enc, days=TEST_DAYS, standardize=False, Min_Max=True)
    testing_data.generate_naive_dataset(concat=False, one_hot=True)
    return training_data, validating_data, testing_data

def train_model(training_data, validating_data, testing_data):
    config = build_model(training_data, MODEL_FILE)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_model(sess, saver, config, reload=RELOAD)
        train(sess, config, training_data, saver)
        print("========== Start validating ==========")
        validation(sess, config,
                   validating_data['input']['BI'],
                   validating_data['input']['tax'],
                   validating_data['input']['weather'],
                   validating_data['label'])

        print("========== Start testing ==========")
        validation(sess, config,
                   testing_data.naive_X['BI'],
                   testing_data.naive_X['tax'],
                   testing_data.naive_X['weather'],
                   testing_data.naive_y)

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
    correct_prediction = tf.equal(tf.argmax(config['output'], 1), tf.argmax(config['input']['group_label'], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Model validating Accuracy: %.2f \n" % sess.run(accuracy, feed_dict={config['input']['BI']: input_BI,
                                                                      config['input']['tax']: input_tax,
                                                                      config['input']['weather']: input_weather,
                                                                      config['input']['group_label']: label_y,
                                                                        config['keep_prob']: 1.0,
                                                                        config['batch_size']: len(input_BI)}))

def init_config():
    global TRAINING_EPOCHS, BATCH_SIZE, DISPLAY_STEP, TEST_USERS, TEST_DAYS
    TRAINING_EPOCHS = int(MODEL_CONFIG['train']['training_epochs'])
    BATCH_SIZE = int(MODEL_CONFIG['train']['batch_size'])
    DISPLAY_STEP = int(MODEL_CONFIG['train']['display_step'])
    TEST_USERS = ast.literal_eval(MODEL_CONFIG['train']['test_users'])
    if MODEL_CONFIG['train']['test_users'] == 'None' or len(TEST_USERS) == 0:
        TEST_USERS = None
        TEST_DAYS = int(MODEL_CONFIG['train']['test_days'])

def parse_args():
    global MODEL_CONFIG, DATA_PATH, RELOAD, MODEL_FILE
    parser = argparse.ArgumentParser()
    required_arguments = parser.add_argument_group('require_arguments')
    optional_arguments = parser.add_argument_group('optional_arguments')

    required_arguments.add_argument("-m", "--model-config", help="model config file", required=True)

    optional_arguments.add_argument("-r", "--reload-model", help="reload model", action='store_true')

    args = parser.parse_args()
    RELOAD = args.reload_model
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