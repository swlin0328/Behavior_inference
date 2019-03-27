import subprocess
from lib.util.convert2json import dict2json
import pandas as pd

def cluster_num_evaluate():
	process = subprocess.Popen(['Rscript', 'lib/clustering/clustering_num_evaluation.R'], shell=True)
	process.wait()
	n_clusters = pd.read_csv(r'../data/cluster_num.csv', header=0).loc[:, 'x'][0]
	result = {'Cluster_num': int(n_clusters)}
	return dict2json(result, 'Cluster_num')
