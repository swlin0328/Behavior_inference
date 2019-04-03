#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from ..main.lib.model.branch_model import build_model
from ..main.lib.util.dataset import dataset
from ..main.lib.util.preprocess import generate_test_users
from sklearn.metrics import f1_score, accuracy_score
from ..main.lib.config.config import DATA_PATH
from ..main.lib.db.dataset import sql4Dataset
from ..main.lib.db.evaluation import sql4Evaluation
from ..main.lib.util.validation import tf_validation as validation
import copy


def start(config, db_conn):
    testing_dataset, temp_dataset = init_dataset(config, db_conn)
    test_acc, test_f1 = model_evaluate(testing_dataset, temp_dataset, config)

    sql4metrics = sql4Evaluation(config['model_name'], -1, sql_conn=db_conn)
    sql4metrics.save2sql(test_f1, test_acc)


def init_dataset(config, db_conn):
    print("\n preparing testing dataset...")
    sql_dataset = sql4Dataset(dataset_name=config['model_name'], sql_conn=db_conn)
    one_hot_enc = sql_dataset.load_one_hot_from_sql()
    test_users = generate_test_users(config['test_users_per_group'])

    testing_data = dataset(config, test_users=test_users, mode='test', enc=one_hot_enc, standardize=False, min_max=True)
    testing_data.generate_naive_dataset(is_concat=False, is_one_hot=True)
    temp_data = {'user': testing_data.naive_X['user'],
                 'tax': testing_data.naive_X['tax'],
                 'weather': testing_data.naive_X['weather'],
                 'group_label': testing_data.naive_y}
    return testing_data, temp_data


def model_evaluate(testing_dataset, temp_dataset, req_config):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model_config = build_model(temp_dataset, req_config)
        init_model(sess, model_config)
        print("========== Start evaluation ==========")
        test_acc, test_f1 = validation(sess, model_config,
                                       testing_dataset.naive_X['user'],
                                       testing_dataset.naive_X['tax'],
                                       testing_dataset.naive_X['weather'],
                                       testing_dataset.naive_y)
        return test_acc, test_f1


def init_model(sess, config):
    sess.run(config['init'])
    saver = tf.train.Saver()
    path = DATA_PATH + 'result/model/' + config['model_name'] + '.ckpt'
    saver.restore(sess, path)
    print("=== Model restored ===")


if __name__ == '__main__':
    start()
