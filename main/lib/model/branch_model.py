#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from configobj import ConfigObj
import tensorflow as tf


def build_model(training_dataset, config_file):
    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        model_config = ConfigObj(config_file)
        keep_prob = tf.placeholder(tf.float32, [])
        batch_size = tf.placeholder(tf.int32, [])
        learning_rate = float(model_config['train']['learning_rate'])

        data_layer = init_input_tensor(training_dataset)
        hidden_layer = init_hidden_tensor(data_layer, keep_prob, batch_size)
        output_layer = init_output_tensor(data_layer, hidden_layer)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=data_layer['group_label'], logits=output_layer))
        correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(data_layer['group_label'], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = optimizer.compute_gradients(loss=cost)
        train_op = optimizer.apply_gradients(gradients)
        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        for gradient, variable in gradients:
            tf.summary.histogram("gradients/" + variable.name, l2_norm(gradient))
            tf.summary.histogram("variables/" + variable.name, l2_norm(variable))

        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)
        init = tf.global_variables_initializer()
        config = {'input': data_layer, 'hidden': hidden_layer, 'output': output_layer, 'optimizer': train_op, 'init': init,
                'cost': cost, 'acc': accuracy, 'model_name': model_config['model']['name'], 'keep_prob': keep_prob,
                'batch_size': batch_size}
        return config


def init_input_tensor(training_dataset):
    input_tensor = {}

    batch_x, batch_ys = training_dataset.next_train_batch(1)
    training_data = batch_x
    training_data['group_label'] = batch_ys
    for input_id in training_data.keys():
        with tf.variable_scope(name_or_scope='input_layer_' + input_id, reuse=False):
            input_tensor[input_id] = tf.placeholder("float", [None, len(training_data[input_id][0])], name=input_id)
    return input_tensor


def init_hidden_tensor(data_layer, keep_prob, batch_size):
    hidden_layer = {}
    for input_id in data_layer.keys():
        if input_id == 'group_label':
            continue

        if input_id == 'BI':
            with tf.variable_scope(name_or_scope='hidden_' + input_id, reuse=False):
                fc_out = tf.layers.dense(data_layer[input_id], 20, activation=tf.nn.relu)
                hidden_layer[input_id] = tf.layers.dense(fc_out, 15, activation=tf.nn.relu)
        else:
            with tf.variable_scope(name_or_scope='hidden_' + input_id, reuse=False):
                input_x = tf.expand_dims(data_layer[input_id], -1)
                conv_out = conv_layer(input_x, filters=8, kernel_size=5, name="conv")

                stacked_rnn = [lstm_cell(5, keep_prob) for _ in range(1)]
                mlstm_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
                init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(mlstm_cell, conv_out, initial_state=init_state, time_major=False)
                hidden_layer[input_id] = tf.layers.flatten(outputs[:, -1, :])

    with tf.variable_scope(name_or_scope='concat_layer', reuse=False):
        hidden_output = [out for out in hidden_layer.values()]
        concat_layer = tf.concat(hidden_output, 1)
        output_class = int(data_layer['group_label'].shape[1])
        hidden_layer['concat_layer'] = fc_layer(concat_layer, output_class, name='fc_concat')
        return hidden_layer


def init_output_tensor(data_layer, hidden_layer):
    with tf.variable_scope(name_or_scope='output_layer', reuse=False):
        output_class = int(data_layer['group_label'].shape[1])
        concat_num = int(hidden_layer['concat_layer'].shape[1])
        output = tf.Variable(tf.random_normal([concat_num, output_class]))
        bias_output = tf.Variable(tf.random_normal([output_class]))
        output_layer = tf.matmul(hidden_layer['concat_layer'], output) + bias_output
        return output_layer


def lstm_cell(num_cell, keep_prob):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_cell, state_is_tuple=True)
    return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)


def conv_layer(input, filters, kernel_size=5, pool_size=1, name="conv"):
    with tf.name_scope(name):
        channels_in = int(input.shape[-1])
        w = tf.Variable(tf.truncated_normal([kernel_size, channels_in, filters], stddev=0.1), name='w')
        b = tf.Variable(tf.constant(0.1, shape=[filters], name="b"))
        conv = tf.nn.conv1d(input, w, stride=1, padding='SAME')
        act = tf.nn.relu(conv + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.layers.max_pooling1d(act, pool_size=pool_size, strides=pool_size, padding="SAME")


def fc_layer(input, output_size, name="fc"):
    with tf.name_scope(name):
        input_size = int(input.shape[-1])
        w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name='w')
        b = tf.Variable(tf.constant(0.1, shape=[output_size], name="b"))
        act = tf.nn.sigmoid(tf.matmul(input, w) + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act