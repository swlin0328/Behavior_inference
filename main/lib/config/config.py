#!/usr/bin/python
# -*- coding: utf-8 -*-
from keras import backend as K
import tensorflow as tf
import configparser
import warnings
import ast

JSON_DIR = 'inference/data/json/'

DIR_FILE = 'inference/data/config/dir_path.ini'
DIR_CONFIG = configparser.ConfigParser()
DIR_CONFIG.read(DIR_FILE)

ATTR_FILE = 'inference/data/config/infer_attr.ini'
ATTR_CONFIG = configparser.ConfigParser()
ATTR_CONFIG.read(ATTR_FILE)

DATA_PATH = DIR_CONFIG['DIR']['DATA_DIR']
WEATHER_ATTR = ast.literal_eval(ATTR_CONFIG['weather']['attr'])
TAX_ATTR = ast.literal_eval(ATTR_CONFIG['tax']['attr'])

def set_tf_session(config):
    warnings.filterwarnings("ignore")
    K.clear_session()
    if config['ip_config'] is not None and config['ip_config'] != '':
        tf_config = tf.ConfigProto()
        tf_config.log_device_placement = bool(config['log_device'])
        sess = tf.Session(config['ip_config'], config=tf_config)
        K.set_session(sess)

def set_attr_config(config):
    config_path = 'inference/data/config/infer_attr.ini'
    parser = configparser.ConfigParser()
    parser.read(config_path)
    weather_attr = ['Temperature']
    tax_attr = ['平均數']
    if config['humidity']:
        weather_attr.append('Humidity')
    if config['tax_median']:
        tax_attr.append('中位數')

    parser.set('weather', 'attr', str(weather_attr))
    parser.set('tax', 'attr', str(tax_attr))
    with open(config_path, 'w+') as configfile:
        parser.write(configfile)
