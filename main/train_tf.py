#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import configparser
import tensorflow as tf
from lib.util.build_tf_model import build_model
from lib.util.dataset import dataset
import ast
import warnings
warnings.filterwarnings("ignore")

MODEL_FILE = '../data/config/tf_model_config.ini'
MODEL_CONFIG = configparser.ConfigParser()
MODEL_CONFIG.read(MODEL_FILE)

TRAINING_EPOCHS = None
BATCH_SIZE = None
DISPLAY_STEP = None
TEST_USERS = None
TEST_DAYS = 30

def main():
    init_config()
    training_data, validating_data, testing_data = init_dataset()
    train_model(training_data, validating_data, testing_data)

def init_dataset():
    print("\n preparing training dataset...")
    training_data = dataset(test_users=TEST_USERS, days=TEST_DAYS, standardize=True)
    training_data.split_dataset(concat=False, one_hot=True)

    validating_X, validating_y = training_data.get_non_concate_dataset(training_data.X_test, training_data.y_test)
    validating_data = {'input':validating_X, 'label':validating_y}
    scaler, one_hot_enc = training_data.get_dataprocessor()

    print("\n preparing testing dataset...")
    testing_data = dataset(test_users=TEST_USERS, mode='test', enc=one_hot_enc, days=TEST_DAYS, scaler=scaler, standardize=True)
    testing_data.generate_naive_dataset(concat=False, one_hot=True)
    return training_data, validating_data, testing_data

def train_model(training_data, validating_data, testing_data):
    with tf.Session() as sess:
        config = build_model()
        sess.run(config['init'])
        train(sess, config, training_data)
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

def train(sess, config, training_data):
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/TensorBoard', sess.graph)
    print("\n ========== Start training ========== \n")
    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0
        total_batch = int(training_data.num_examples() / BATCH_SIZE)
        for i in range(total_batch):
            batch_x, batch_ys = training_data.next_train_batch(BATCH_SIZE)
            ___, summary = sess.run([config['optimizer'], merged], feed_dict={config['input']['x_BI']: batch_x['BI'],
                                                                    config['input']['x_tax']: batch_x['tax'],
                                                                    config['input']['x_weather']: batch_x['weather'],
                                                                    config['input']['y']: batch_ys})
            avg_cost += sess.run(config['cost'], feed_dict={config['input']['x_BI']: batch_x['BI'],
                                                  config['input']['x_tax']: batch_x['tax'],
                                                  config['input']['x_weather']: batch_x['weather'],
                                                  config['input']['y']: batch_ys})
        if epoch % DISPLAY_STEP == 0:
            train_writer.add_summary(summary, epoch)
            print('Epoch: %04d cost= %.9f' % ((epoch + 1), avg_cost))

    train_writer.close()
    print("Training phase finished... \n")

def validation(sess, config, input_BI, input_tax, input_weather, label_y):
    correct_prediction = tf.equal(tf.argmax(config['output'], 1), tf.argmax(config['input']['y'], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Model validating Accuracy: %.2f \n" % sess.run(accuracy, feed_dict={config['input']['x_BI']: input_BI,
                                                                      config['input']['x_tax']: input_tax,
                                                                      config['input']['x_weather']: input_weather,
                                                                      config['input']['y']: label_y}))

def init_config():
    global TRAINING_EPOCHS, BATCH_SIZE, DISPLAY_STEP, TEST_USERS, TEST_DAYS
    TRAINING_EPOCHS = int(MODEL_CONFIG['train']['training_epochs'])
    BATCH_SIZE = int(MODEL_CONFIG['train']['batch_size'])
    DISPLAY_STEP = int(MODEL_CONFIG['train']['display_step'])
    TEST_USERS = ast.literal_eval(MODEL_CONFIG['train']['test_users'])
    if MODEL_CONFIG['train']['test_users'] == 'None' or len(TEST_USERS)==0:
        TEST_USERS = None
        TEST_DAYS = int(MODEL_CONFIG['train']['test_days'])

if __name__ == '__main__':
    main()
