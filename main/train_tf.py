#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from ..main.lib.model.branch_model import build_model
from ..main.lib.util.dataset import dataset
from ..main.lib.util.preprocess import generate_test_users
from ..main.lib.config.config import DATA_PATH
from ..main.lib.db.dataset import sql4Dataset
from ..main.lib.util.validation import tf_validation as validation
from ..main.lib.db.dnn_model import sql4Tensorflow
from sklearn.metrics import f1_score, accuracy_score


def start(req_config, db_conn):
    training_data, validating_data, temp_dataset = init_dataset(req_config, db_conn)
    validate_acc, validate_f1 = train_pipeline(training_data, validating_data, temp_dataset, req_config, db_conn)
    print_metrics(validate_acc, validate_f1)


def init_dataset(req_config, db_conn):
    test_users = generate_test_users(req_config['test_users_per_group'])
    print("\n preparing training dataset...")
    training_data = dataset(req_config, test_users=test_users, standardize=False, min_max=True)
    training_data.generate_naive_dataset(is_concat=False, is_one_hot=True)
    training_data.split_dataset()
    print("\n preparing validating dataset...")
    validating_X, validating_y = training_data.get_non_concate_dataset(training_data.X_test, training_data.y_test)
    validating_data = {'input': validating_X, 'label': validating_y}
    print("\n init model shape...")
    batch_x, batch_ys = training_data.next_train_batch(1)
    temp_dataset = batch_x
    temp_dataset['group_label'] = batch_ys

    save_one_hot_enc(training_data, req_config, db_conn)
    return training_data, validating_data, temp_dataset


def save_one_hot_enc(training_data, config, db_conn):
    one_hot_enc = training_data.get_dataprocessor()

    sql_dataset = sql4Dataset(dataset_name=config['model_name'], sql_conn=db_conn)
    sql_dataset.save2sql(one_hot=one_hot_enc)


def train_pipeline(training_data, validating_data, temp_data, req_config, db_conn):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.log_device_placement = bool(req_config['log_device'])
    with tf.Session(req_config['ip_config'], config=tf_config) as sess:
        model_config = build_model(temp_data, req_config)
        init_model(sess, model_config, reload=req_config['reload_model'])
        step = train_model(sess, model_config, training_data, req_config)
        print("========== Start validating ==========")
        validate_acc, validate_f1 = validation(sess, model_config,
                                               validating_data['input']['user'],
                                               validating_data['input']['tax'],
                                               validating_data['input']['weather'],
                                               validating_data['label'])
        save_model(sess, req_config, valid_acc=validate_acc, num_step=step, db_conn=db_conn)
        return validate_acc, validate_f1


def train_model(sess, model_config, training_data, req_config):
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/TensorBoard', sess.graph)
    print("\n ========== Start training ========== \n")
    step = 0
    for epoch in range(int(req_config['training_epochs'])):
        avg_cost = 0
        batch_size = int(req_config['batch_size'])
        total_batch = int(training_data.num_examples() / batch_size)
        for i in range(total_batch):
            step = step + 1
            batch_x, batch_ys = training_data.next_train_batch(batch_size)
            ___, summary = sess.run([model_config['optimizer'], merged],
                                    feed_dict={model_config['input']['user']: batch_x['user'],
                                               model_config['input']['tax']: batch_x['tax'],
                                               model_config['input']['weather']: batch_x['weather'],
                                               model_config['input']['group_label']: batch_ys,
                                               model_config['keep_prob']: 0.8,
                                               model_config['batch_size']: batch_size})
            avg_cost += sess.run(model_config['cost'], feed_dict={model_config['input']['user']: batch_x['user'],
                                                                  model_config['input']['tax']: batch_x['tax'],
                                                                  model_config['input']['weather']: batch_x['weather'],
                                                                  model_config['input']['group_label']: batch_ys,
                                                                  model_config['keep_prob']: 0.8,
                                                                  model_config['batch_size']: batch_size})
        if epoch % int(req_config['display_step']) == 0:
            train_writer.add_summary(summary, epoch)
            print('Epoch: %04d cost= %.9f' % ((epoch + 1), avg_cost))
    train_writer.close()
    print("Training phase finished... \n")
    return step


def init_model(sess, config, reload):
    sess.run(config['init'])
    if reload == 'True':
        saver = tf.train.Saver()
        path = DATA_PATH + 'result/model/' + config['model_name'] + '.ckpt'
        saver.restore(sess, path)
        print("=== Model restored ===")


def save_model(sess, config, valid_acc, num_step, db_conn):
    saver = tf.train.Saver()
    path = DATA_PATH + 'result/model/' + config['model_name'] + '.ckpt'
    saver.save(sess, path)

    num_params = count_parameters()
    sql4model = sql4Tensorflow(model_name=config['model_name'], customer_group=-1, creator=config['user'],
                               sql_conn=db_conn)
    sql4model.save2sql(model_type=config['model_type'], model_params=num_params, valid_acc=valid_acc, step=num_step)
    print("=== Model saved ===")


def print_metrics(validate_acc, validate_f1):
    print('validating accuracy: ', validate_acc)
    print('validating f1_score: ', validate_f1)
    print('Validating phase finished... \n')


def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    return total_parameters


if __name__ == '__main__':
    start()
