from flask import Flask
from flask import request
import warnings

import lib
from lib.preprocess.dataset_description import list_dataset
from lib.preprocess.dataset_description import dataset_statistics
from lib.preprocess.correlation_analysis import plot_correlation_analysis
from lib.preprocess.dataset_preprocess import load_data
from lib.preprocess.dataset_preprocess import data_cleaning
from lib.preprocess.dataset_preprocess import data_scaling
from lib.preprocess.dataset_preprocess import feature_engineering
from lib.preprocess.dataset_preprocess import one_hot_encoding
from lib.preprocess.dataset_preprocess import dataset_split
from lib.pipeline.dataset_pipeline import pipeline
from lib.clustering.clustering_num_evaluation import cluster_num_evaluate
from lib.clustering.clustering_analysis import cluster_training
from lib.clustering.clustering_analysis import group_analysis
from lib.util.convert2json import str2object

app = Flask(__name__)

@app.route('/data_Analysis/dataset_list', methods=['GET'])
def dataset_list():
	return list_dataset()

@app.route('/data_Analysis/head_N_data', methods=['POST'])
def head_N_data():
	num_data = 10
	dataset_name = request.form['dataSet']
	dataset = dataset_statistics(dataset_name)
	return dataset.list_data(num_data)

@app.route('/data_Analysis/data_statistics', methods=['POST'])
def data_statistics():
	dataset_name = request.form['dataSet']
	attribute = str2object(request.form['attribute'])
	statistics = str2object(request.form['statistics'])
	dataset = dataset_statistics(dataset_name)
	return dataset.describe_dataset(attr=attribute, analysis=statistics)

@app.route('/data_Analysis/correlation_analysis', methods=['POST'])
def correlation_analysis():
	dataset_name = request.form['dataSet']
	attribute = str2object(request.form['target_attr'])
	method= request.form['method']
	fig_size = int(request.form['fig_size'])
	return plot_correlation_analysis(target_attr=attribute, filename=dataset_name, method=method, size=fig_size)

@app.route('/data_Preprocess/load_dataset', methods=['POST'])
def load_dataset():
	dataset_name = request.form['dataSet']
	num_data = int(request.form['num_data'])
	return load_data(dataset=dataset_name, num_data=num_data)

@app.route('/data_Preprocess/data_Cleaning', methods=['POST'])
def data_Cleaning():
	dataset_name = request.form['dataSet']
	attr = str2object(request.form['attr'])
	method = request.form['method']
	return data_cleaning(dataset_name, attr=attr, method=method)

@app.route('/data_Preprocess/feature_Scaling', methods=['POST'])
def feature_Scaling():
	dataset_name = request.form['dataSet']
	scaling_method = str2object(request.form['scaling'])
	return data_scaling(dataset_name, scaling_method)

@app.route('/data_Preprocess/feature_Engineering', methods=['POST'])
def feature_Engineering():
	dataset_name = request.form['dataSet']
	feature_1 = request.form['feature_1']
	feature_2 = request.form['feature_2']
	operator = request.form['operator']
	return feature_engineering(dataset_name, feature_1, feature_2, operator)

@app.route('/data_Preprocess/one_hot_Encoding', methods=['POST'])
def one_hot_Encoding():
	dataset_name = request.form['dataSet']
	feature = request.form['feature']
	return one_hot_encoding(dataset_name, feature)

@app.route('/data_Preprocess/dataset_Splitting', methods=['POST'])
def dataset_Splitting():
	dataset_name = request.form['dataSet']
	stratify = int(request.form['stratify'])
	attribute = request.form['attribute']
	return dataset_split(dataset_name, stratify, attribute)

@app.route('/data_Preprocess/dataset_Pipeline', methods=['POST'])
def dataset_Pipeline():
	dataset_name = request.form['dataSet']
	pipe = request.form['pipeline']
	return pipeline(dataset_name, pipeline=pipe)

@app.route('/clustering/cluster_num_evaluation', methods=['GET'])
def cluster_num_evaluation():
	return cluster_num_evaluate()

@app.route('/clustering/cluster_model_training', methods=['POST'])
def cluster_model_training():
	dataset_name = request.form['dataSet']
	n_clusters = int(request.form['n_clusters'])
	fig_size = int(request.form['fig_size'])
	return cluster_training(dataset_name, n_clusters, fig_size)

@app.route('/clustering/cluster_analysis', methods=['POST'])
def cluster_analysis():
	model_result = request.form['model_result']
	fig_size = int(request.form['fig_size'])
	return group_analysis(fig_size, model_result)

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	app.run(host='111.222.33.44', port=2222, debug=False, threaded=True)
