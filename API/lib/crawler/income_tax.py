import os
import urllib
import configparser
import pandas as pd

class income_tax():
	def __init__(self, config_file='../data/config/income_tax.ini'):
		self.config = configparser.ConfigParser()
		self.config.read(config_file)
		self.DATA_DIR = '../data/'
		self.TAX_DIR = self.DATA_DIR + 'income_tax/'
		if not os.path.isdir(self.TAX_DIR):
			os.makedirs(self.TAX_DIR)

	def download_data(self):
		for city_name, city_code in self.config['city'].items():
			duration = int(self.config['duration']['latest']) - 100
			for year in range(duration):
				year = str(year + 101)
				urllib.request.urlretrieve(
					r'https://ws.fia.gov.tw/001/upload/ias/ias' + year + '/' + year + '_165-' + city_code + '.csv',
					self.TAX_DIR + city_name + '_' + year + '.csv')

	def integrate2csv(self):
		result = pd.DataFrame()
		for filename in os.listdir(self.TAX_DIR):
			file_info = filename.split('.csv')[0].split('_')
			city = file_info[0]
			year = file_info[1]
			df = pd.read_csv(self.TAX_DIR + filename, encoding='utf_8_sig')
			df.insert(loc=0, column='年度', value=year)
			df.insert(loc=0, column='縣市', value=city)
			result = result.append(df, ignore_index=True)

		result.to_csv(self.DATA_DIR + '綜合所得稅各縣市統計分析表.csv', encoding='utf_8_sig', index=False)

	def start(self):
		self.download_data()
		self.integrate2csv()