# System Imports
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import warnings

#inference
from main import train_binary_keras, test_binary_keras
from main import train_scikit, test_scikit
from main.lib.db.db_util import sql4DB
from main.lib.util.convert2json import df2json

from config.config import DB_PATH

app = Flask(__name__)
app.config['SECRET_KEY'] = ''
app.config['SQLALCHEMY_DATABASE_URI'] = DB_PATH
db = SQLAlchemy(app)

@app.route("/")
@app.route("/home")
def home():
	return render_template('home.html')

@app.route("/inference/train", methods=['POST'])
def inference_training():
	config = request.form.to_dict()
	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()
	if config['model_type'] == 'DNN':
		train_binary_keras.start(config, db_conn)
	elif config['model_type'] == 'SVM':
		train_scikit.start(config, db_conn)
	return "Successful"


@app.route("/inference/test", methods=['POST'])
def inference_testing():
	db_conn = db.create_engine(DB_PATH, deprecate_large_types=True).raw_connection()
	config = request.form.to_dict()

	query = sql4DB(config, db_conn)
	model_id = query.search_model()
	model_type = query.read_model_type(model_id)

	if model_type == 'DNN':
		test_binary_keras.start(config, db_conn)
	elif model_type == 'SVM':
		test_scikit.start(config, db_conn)
	return "Successful"


@app.route("/inference/upload_model", methods=['POST'])
def inference_upload_model():
	config = request.form.to_dict()
	if request.files['file'] is not None:
		path = r'./inference/taipower_inference/lib/model/'
		file = request.files['file']
		file.save(path + config['model_name'] + '.py')
	return "Successful"


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


if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	app.run(host='111.111.111.11', port=5000, debug=False, threaded=True)
