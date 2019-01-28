#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from configobj import ConfigObj
import tensorflow as tf

def build_model(training_dataset, config_file):
    model_config = ConfigObj(config_file)
    keep_prob = tf.placeholder(tf.float32, [])
    batch_size = tf.placeholder(tf.int32, [])
    learning_rate = float(model_config['train']['learning_rate'])

    input_layer = init_input_tensor(training_dataset)
    hidden_layer = init_hidden_tensor(model_config, input_layer)
    output_layer = init_output_tensor(input_layer, hidden_layer)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_layer['group_label'], logits=output_layer))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    tf.summary.scalar('loss', cost)
    init = tf.global_variables_initializer()
    config = {'input': input_layer, 'hidden': hidden_layer, 'output': output_layer, 'optimizer': optimizer,
              'init': init, 'cost': cost, 'model_name': model_config['model']['name'], 'keep_prob': keep_prob,
              'batch_size': batch_size}
    return config

def init_input_tensor(training_dataset):
    input_tensor = {}

    batch_x, batch_ys = training_dataset.next_train_batch(1)
    training_data = batch_x
    training_data['group_label'] = batch_ys
    for input_id in training_data.keys():
        input_tensor[input_id] = tf.placeholder("float", [None, len(training_data[input_id][0])], name=input_id)
    return input_tensor

def init_hidden_tensor(model_config, input_layer):
    hidden_layer = {}
    hidden_structure = model_config['structure'].keys()
    num_hidden_layer = len(hidden_structure)

    for layer_idx, layer_key in enumerate(hidden_structure):
        layer = {}
        neuron_in_layer = 0
        for layer_id in model_config['structure'][layer_key].keys():
            if layer_id not in model_config['structure'][layer_key]['layer_input'].keys():
                continue
            input_id = model_config['structure'][layer_key]['layer_input'][layer_id]
            hidden_num = int(model_config['structure'][layer_key][layer_id])

            if layer_idx == 0:
                input_num = int(input_layer[input_id].shape[1])
                tf_last_layer = input_layer[input_id]
            else:
                last_layer_key = 'layer_' + str(layer_idx)
                input_num = int(model_config['structure'][last_layer_key][input_id])

                last_layer_id = hidden_structure[layer_idx - 1]
                tf_last_layer = hidden_layer[last_layer_id][input_id]

            hidden = tf.Variable(tf.random_normal([input_num, hidden_num]))
            bias = tf.Variable(tf.random_normal([hidden_num]))
            layer[layer_id] = tf.nn.sigmoid(tf.add(tf.matmul(tf_last_layer, hidden), bias))
            neuron_in_layer = neuron_in_layer + hidden_num

        hidden_layer[layer_key] = layer
        if layer_idx + 2 == num_hidden_layer:
            hidden_output = [out for out in layer.values()]
            concat_layer = tf.concat(hidden_output, 1)

            concat_num = int(model_config['structure']['concat_layer']['concat_neuron'])
            w = tf.Variable(tf.random_normal([neuron_in_layer, concat_num]))
            last_bias_layer = tf.Variable(tf.random_normal([concat_num]))
            concat_layer = tf.nn.sigmoid(tf.add(tf.matmul(concat_layer, w), last_bias_layer))
            hidden_layer['concat_layer'] = concat_layer
            break
    return hidden_layer

def init_output_tensor(input_layer, hidden_layer):
    output_class = int(input_layer['group_label'].shape[1])
    concat_num = int(hidden_layer['concat_layer'].shape[1])
    output = tf.Variable(tf.random_normal([concat_num, output_class]))
    bias_output = tf.Variable(tf.random_normal([output_class]))
    output_layer = tf.matmul(hidden_layer['concat_layer'], output) + bias_output
    return output_layer