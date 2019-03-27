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

def set_tf_session(request_config):
    warnings.filterwarnings("ignore")
    K.clear_session()
    tf_config = tf.ConfigProto()
    tf_config.log_device_placement = True
    if request_config['ip_config'] is not None and request_config['ip_config'] != '':
        sess = tf.Session('grpc://' + request_config['ip_config'], config=tf_config)
        K.set_session(sess)
