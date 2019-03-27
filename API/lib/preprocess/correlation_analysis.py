import seaborn as sns
import pandas as pd
from lib.util.convert2json import to_json
import math
import matplotlib.pyplot as plt

DATA_DIR = r'../data/'
RESULT_DIR = r'../data/result/'

def distplot(df, sns_obj, size):
	df_cols = df.columns.values.tolist()
	num_col = 2
	num_row = math.ceil(len(df_cols)/2)
	fig, axarr = plt.subplots(nrows=num_row, ncols=num_col, figsize=(size, size))
	for idx, col_name in enumerate(df_cols):
		col_idx = idx % num_col
		row_idx = math.floor(idx/2)
		if num_row > 1:
			sns_obj.distplot(df[col_name], ax=axarr[row_idx][col_idx])
		else:
			sns_obj.distplot(df[col_name], ax=axarr[col_idx])

	return fig

def pairplot(df, sns_obj, size):
	sns_plot = sns_obj.pairplot(df, height=size)
	return sns_plot

plot_method = {
	'histogram': distplot,
	'pair_plot': pairplot
}

def plot_correlation_analysis(target_attr=['Max_load', 'Min_load'], filename='for_cluster_with_label', method='pair_plot', size=12):
	""" Plot the correlation analysis, and saved with the svg format
	Args :
		filename (str) : The filename of the source data
		cols (int) : Number of columns of the subplot figure.

	Return :
		json_data {key(str), value}: The key/value pairs of json, and the value is the base64_string
	"""
	file_name = DATA_DIR + filename + '.csv'
	svg_filename = 'correlation_analysis'
	df = pd.read_csv(file_name, header=0)

	sns.set(style='whitegrid', context='notebook')
	sns.set(font_scale=1.2)
	sns_plot = plot_method[method](df[target_attr], sns, size)

	sns_plot.savefig(RESULT_DIR + svg_filename + '.svg', format="svg")

	target_file = {'data_correlation': RESULT_DIR + svg_filename + '.svg'}
	return to_json(target_file, svg_filename)