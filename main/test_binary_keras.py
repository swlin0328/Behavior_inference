#!/usr/bin/python
# -*- coding: utf-8 -*-
from ..main.lib.util.dataset import dataset
from ..main.lib.util.validation import anomaly_metrics_validation
from ..main.lib.util.preprocess import get_num_group, generate_test_users
from ..main.lib.config.config import set_tf_session
from ..main.lib.db.dnn_model import sql4Keras
from ..main.lib.db.dataset import sql4Dataset
from ..main.lib.db.evaluation import sql4Evaluation


def start(config, db_conn):
    num_group = get_num_group()
    testing_dataset = init_dataset(config, db_conn=db_conn)
    final_test(testing_dataset, num_group, config, db_conn)


def init_dataset(config, db_conn=None):
    print("\n preparing testing dataset...")
    test_users = generate_test_users(config['test_users_per_group'])
    test_dataset = dataset(config, test_users=test_users, mode='test', standardize=False, min_max=True)
    test_dataset = pca_dataset(test_dataset, config, db_conn)
    return test_dataset


def pca_dataset(test_dataset, config, db_conn):
    sql_dataset = sql4Dataset(dataset_name=config['model_name'], sql_conn=db_conn)
    pca = sql_dataset.load_pca_from_sql()
    test_dataset.pca_naive_X(pca_transducer=pca)
    return test_dataset


def final_test(testing_dataset, num_group, config, db_conn):
    for group_idx in range(num_group):
        model, mse_threshold = init_model(config, group_idx, db_conn)
        acc, f1_score = anomaly_metrics_validation(model, mse_threshold, testing_dataset, group_idx)

        sql4metrics = sql4Evaluation(config['model_name'], group_idx, sql_conn=db_conn)
        sql4metrics.save2sql(f1_score, acc)


def init_model(config, target_label, db_conn):
    sql4model = sql4Keras(model_name=config['model_name'], customer_group=target_label, sql_conn=db_conn)
    model = sql4model.load_model_from_sql()
    mse_threshold = sql4model.load_threshold_from_sql()
    print(model.summary())
    return model, mse_threshold


if __name__ == '__main__':
    start()