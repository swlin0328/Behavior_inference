#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.metrics import f1_score, accuracy_score
from ..main.lib.util.dataset import dataset
from ..main.lib.util.preprocess import generate_test_users
from ..main.lib.db.svm_model import sql4SVM
from ..main.lib.db.dataset import sql4Dataset
from ..main.lib.db.evaluation import sql4Evaluation
import warnings


def start(config, db_conn):
    warnings.filterwarnings("ignore")
    testing_data = init_dataset(config, db_conn)
    test_pipeline(testing_data, config, db_conn)


def init_dataset(config, db_conn):
    test_users = generate_test_users(config['test_users_per_group'])
    print("\n preparing testing dataset...")
    testing_data = dataset(config, test_users=test_users, mode='test', standardize=False, min_max=True)
    testing_data = pca_naive_dataset(testing_data, config, db_conn)
    return testing_data


def pca_naive_dataset(testing_data, config, db_conn):
    sql_dataset = sql4Dataset(dataset_name=config['model_name'], sql_conn=db_conn)
    pca = sql_dataset.load_pca_from_sql()
    testing_data.pca_naive_X(pca_transducer=pca)
    return testing_data


def test_pipeline(testing_data, config, db_conn):
    model = init_model(config, db_conn)

    predict_y = model.predict(testing_data.naive_X)
    test_acc = accuracy_score(testing_data.naive_y, predict_y)
    test_f1 = f1_score(testing_data.naive_y, predict_y, average='macro')
    print('Accuracy for testing data: %.3f' % test_acc)
    print('F1_score for testing data: %.3f' % test_f1)

    sql4metrics = sql4Evaluation(config['model_name'], -1, sql_conn=db_conn)
    sql4metrics.save2sql(test_f1, test_acc)


def init_model(config, db_conn):
    sql4model = sql4SVM(model_name=config['model_name'], customer_group=-1, sql_conn=db_conn)
    model = sql4model.load_model_from_sql()
    return model


if __name__ == '__main__':
    main()