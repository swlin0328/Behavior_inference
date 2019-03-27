import base64
import json
import ast
from ..config.config import JSON_DIR

def to_json(dict, json_filename):
	""" convert the target file to json format.
	Args :
		dict  {key(str), value} : The key/value pairs for json, the value is the target data, which could be the filename(str) or data (object, bytes).
		json_filename (str) : The target filename for json.

	Return :
		json_data {key(str), value}: The key/value pairs of json, and the value is the base64_string encoded with the target file.
	"""
	ENCODING = 'utf-8'
	JSON_NAME = JSON_DIR + json_filename + '.json'
	raw_data = {}
	for key, value in dict.items():
		if isinstance(value, str):
			FILE_NAME = value
			with open(FILE_NAME, 'rb') as target_file:
				byte_content = target_file.read()
		elif isinstance(value, bytes):
			byte_content = value
		else:
			byte_content = bytes(value)

		base64_bytes = base64.b64encode(byte_content)
		base64_string = base64_bytes.decode(ENCODING)
		raw_data.setdefault(key, base64_string)

	json_data = json.dumps(raw_data, indent=2)
	with open(JSON_NAME, 'w') as json_file:
		json_file.write(json_data)
	return json_data

def dict2json(dict, json_filename):
	JSON_NAME = JSON_DIR + json_filename + '.json'

	json_data = json.dumps(dict)
	with open(JSON_NAME, 'w') as json_file:
		json_file.write(json_data)
	return json_data

def df2json(df, json_filename, orient='records'):
	""" convert the dataframe to the json format.
	Args :
		df (pandas.DataFrame) : The target dataframe that could be converted to json file by pandas.
		json_filename (str) : The target filename for json.
		orient (str) : Format for saving json file, which could be 'records', 'index', and 'columns'.

	Return :
		JSON string.
	"""
	df.to_json(JSON_DIR + json_filename + '.json', orient=orient, force_ascii=False)
	return df.to_json(orient=orient, force_ascii=False)

def decode_json2data(json_filename):
	""" Load the json file, converting the target file to the binary stream.
	Args :
		json_filename (str) : The filename of json.

	Return :
		file_dict {key(str), value}: The key/value pairs of the target data, and the value is the binary stream decoded with the base64_string
	"""
	JSON_NAME = JSON_DIR + json_filename + '.json'

	with open(JSON_NAME, "r") as json_file:
		raw_data = json.load(json_file)

	file_dict = {}
	for key, value in raw_data.items():
		file_base64_string = raw_data[key]
		file_data = base64.b64decode(file_base64_string)
		file_dict.setdefault(key, file_data)

	return file_dict

def convert_json2data(filename, filename_extension=".svg"):
	""" decode data to target format saved in json file
	Args :
		filename : The filename of json.
		extension : The target format converted from the json file.

	Return :
		success (bool) : The state of converting json to file.
	"""
	raw_data = decode_json2data(filename)
	success = False

	for key, value in raw_data.items():
		file_path = JSON_DIR + key + filename_extension
		with open(file_path, 'wb') as data_file:
			data_file.write(value)
			success = True
	return success

def combine_json(filename1, filename2, append_child=None):
	file_in1 = open(JSON_DIR + filename1 + '.json', "r")
	file_in2 = open(JSON_DIR + filename2 + '.json', "r")
	file_out = open(JSON_DIR + 'combine.json', "w")

	json_data1 = json.load(file_in1)
	json_data2 = json.load(file_in2)
	combine = json_data1

	if append_child is None:
		for key, value in json_data2.items():
			combine.setdefault(key, value)
	else:
		combine.setdefault(append_child, json_data2)

	file_out.write(json.dumps(combine))
	file_in1.close()
	file_in2.close()
	file_out.close()
	return json.dumps(combine)

def combine_json_contain_list(json_1, json_2, child_1, child_2):
	combine = {child_1: ast.literal_eval(json_1), child_2: ast.literal_eval(json_2)}
	return dict2json(combine, 'combined_json')

def str2object(string):
	return ast.literal_eval(string)