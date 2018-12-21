#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from configobj import ConfigObj
import tensorflow as tf

MODEL_FILE = '../data/config/tf_model_config.ini'
MODEL_CONFIG = ConfigObj(MODEL_FILE)

def build_model():
    input = {}
    for input_id in MODEL_CONFIG['input'].keys():
        input[input_id] = tf.placeholder("float", [None, MODEL_CONFIG['input'][input_id]], name=input_id)

    model = {}
    model_layers = MODEL_CONFIG['structure'].keys()
    num_hidden_layer = len(model_layers)
    concat_num = 0
    for layer_idx, layer_key in enumerate(model_layers):
        hidden_layer = {}
        last_layer_key = 'layer_' + str(layer_idx)
        neuron_in_layer = 0
        if layer_idx == 0:
            last_layer = MODEL_CONFIG['input']
        else:
            last_layer = MODEL_CONFIG['structure'][last_layer_key]

        for layer_id in MODEL_CONFIG['structure'][layer_key].keys():
            if layer_id not in MODEL_CONFIG['structure'][layer_key]['layer_input'].keys():
                continue
            input_id = MODEL_CONFIG['structure'][layer_key]['layer_input'][layer_id]
            hidden_num = int(MODEL_CONFIG['structure'][layer_key][layer_id])
            input_num = int(last_layer[input_id])

            if layer_idx == 0:
                tf_last_layer = input[input_id]
            else:
                last_layer_id = model_layers[layer_idx-1]
                tf_last_layer = model[last_layer_id][input_id]

            hidden = tf.Variable(tf.random_normal([input_num, hidden_num]))
            bias = tf.Variable(tf.random_normal([hidden_num]))
            hidden_layer[layer_id] = tf.nn.sigmoid(tf.add(tf.matmul(tf_last_layer, hidden), bias))
            neuron_in_layer = neuron_in_layer + hidden_num

        model[layer_key] = hidden_layer
        if layer_idx + 2 == num_hidden_layer:
            hidden_output = [out for out in hidden_layer.values()]
            concat_layer = tf.concat(hidden_output, 1)

            concat_num = int(MODEL_CONFIG['structure']['concat_layer']['concat_neuron'])
            w = tf.Variable(tf.random_normal([neuron_in_layer, concat_num]))
            last_bias_layer = tf.Variable(tf.random_normal([concat_num]))
            concat_layer = tf.nn.sigmoid(tf.add(tf.matmul(concat_layer, w), last_bias_layer))
            model['concat_layer'] = concat_layer
            break

    output_class = int(MODEL_CONFIG['input']['y'])
    output = tf.Variable(tf.random_normal([concat_num, output_class]))
    bias_output = tf.Variable(tf.random_normal([output_class]))
    output_layer = tf.matmul(model['concat_layer'], output) + bias_output
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input['y'], logits=output_layer))
    tf.summary.scalar('loss', cost)

    learning_rate = float(MODEL_CONFIG['train']['learning_rate'])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    config = {'input':input, 'hidden':model, 'output':output_layer, 'optimizer':optimizer, 'init':init, 'cost':cost}
    return config