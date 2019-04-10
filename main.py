# System Imports
from flask import Flask, render_template, url_for, flash, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask import send_file
import warnings
import os

#inference
from inference.main import train_binary_keras, test_binary_keras
from inference.main import train_scikit, test_scikit
from inference.main import train_tf, test_tf
from inference.main.lib.db.db_util import sql4DB
from inference.main.lib.cluster.clustering import clustering
from inference.main.lib.db.download_dataset import DB2dataset
from inference.main.lib.util.convert2json import df2json, dict2json
from inference.main.lib.config.config import set_attr_config

# Configuration
from config.config import DB_PATH

app = Flask(__name__)
app.config['SECRET_KEY'] = ''
app.config['SQLALCHEMY_DATABASE_URI'] = DB_PATH
db = SQLAlchemy(app)
COUNT = 0

@app.route("/inference/train", methods=['POST'])
def inference_training():
	config = request.form.to_dict()
	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()
	if config['model_type'] == 'DNN':
		train_tf.start(config, db_conn)
	elif config['model_type'] == 'Anomaly':
		train_binary_keras.start(config, db_conn)
	elif config['model_type'] == 'SVM':
		train_scikit.start(config, db_conn)

	query = sql4DB(config, db_conn)
	model_id = query.search_model()
	result_df = query.read_model_info(model_id)
	return df2json(result_df, 'train_result', orient='records')


@app.route("/inference/train", methods=['GET'])
def inference_auto_training():
	global COUNT
	config = {
		'model_name': 'svm_test_' + str(COUNT),
		'model_type': 'SVM',
		'pca_dim': 25,
		'gamma': 0.1,
		'C': 10.0,
		'user': 'EE',
		'test_users_per_group': 1}

	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()
	train_scikit.start(config, db_conn)

	query = sql4DB(config, db_conn)
	model_id = query.search_model()
	result_df = query.read_model_info(model_id)
	COUNT = COUNT + 1
	return df2json(result_df, 'train_result', orient='records')

@app.route("/inference/test", methods=['POST'])
def inference_testing():
	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()
	config = request.form.to_dict()

	query = sql4DB(config, db_conn)
	model_id = query.search_model()
	model_type = query.read_model_type(model_id)

	if model_type == 'DNN':
		test_tf.start(config, db_conn)
	elif model_type == 'Anomaly':
		test_binary_keras.start(config, db_conn)
	elif model_type == 'SVM':
		test_scikit.start(config, db_conn)

	result_df = query.read_evaluation_info(model_id)
	return df2json(result_df, 'evaluation_result', orient='records')


@app.route("/inference/upload_model", methods=['POST'])
def inference_upload_model():
	path = r'./inference/lib/model/'
	config = request.form
	dir_path = path + config['model_name']
	if not os.path.isdir(dir_path):
		os.makedirs(dir_path)

	file = request.files['file']
	file.save(dir_path + '/customized_model.py')
	return redirect('http://111.11.111.111:5000/inference/train', code=307)


@app.route("/inference/model_query", methods=['POST'])
def inference_query():
	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()
	config = request.form.to_dict()
	query = sql4DB(config, db_conn)

	if config['query_type'] == 'model':
		result_df = query.read_model_info()
	elif config['query_type'] == 'model_layer':
		result_df = query.read_layer_info()
	elif config['query_type'] == 'dataset':
		result_df = query.read_dataset_info()
	elif config['query_type'] == 'evaluation':
		result_df = query.read_evaluation_info()

	return df2json(result_df, 'query_result', orient='records')


@app.route("/inference/download_dataset", methods=['POST'])
def inference_download():
	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()
	config = request.form.to_dict()
	set_attr_config(config)

	customer_dataset_download = DB2dataset(config, sql_conn=db_conn)
	customer_dataset_download.start()

	customer_dataset_download = clustering(config['cluster_model'], sql_conn=db_conn)
	customer_dataset_download.start()
	result = {'download_state': True}
	return dict2json(result, 'download_state')


@app.route("/inference/model_train_string", methods=['GET'])
def train_info():
	dataset_list = list_dataset()
	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()

	query = sql4DB(sql_conn=db_conn)
	result_df = query.read_cluster_model_info()
	result_df = result_df.drop_duplicates(['model_name'])
	result_df = result_df['model_name']
	model_list = df2json(result_df, 'model_list', orient='records')
	return combine_json_contain_list(dataset_list, model_list, 'dataset', 'model')


@app.route("/inference/model_evaluation_string", methods=['GET'])
def evaluation_info():
	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()
	config = {'query_type': 'model'}

	query = sql4DB(config, db_conn)
	query_string = query.produce_query_string()
	result_df = query.read_data(query_string)
	return df2json(result_df, 'model_list', orient='records')


@app.route("/inference/model_query_string", methods=['POST'])
def query_info():
	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()
	config = request.form.to_dict()

	query = sql4DB(config, db_conn)
	query_string = query.produce_query_string()
	result_df = query.read_data(query_string)
	return df2json(result_df, 'query_result', orient='records')


@app.route("inference/download_template", methods=['Get'])
def download_template():
	try:
		return send_file('./inference/lib/model/keras_autoencoder.py', attachment_filename='template.py')
	except Exception as e:
		return str(e)

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	app.run(host='', port=5000, debug=False, threaded=True)
