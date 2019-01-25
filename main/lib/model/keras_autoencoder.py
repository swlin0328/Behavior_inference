#!/usr/bin/python
# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, LeakyReLU, concatenate, BatchNormalization, PReLU
from keras import regularizers

def build_model(input_shape, encoding_dim=10):
    input_data = Input(shape=[input_shape])
    encoded = Dense(encoding_dim * 4, activation='relu')(input_data)
    encoded = Dense(encoding_dim * 2, activation='relu')(encoded)

    encoded = Dense(encoding_dim, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01))(encoded)

    decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = Dense(encoding_dim * 4, activation='relu')(decoded)
    decoded = Dense(input_shape, activation='relu')(decoded)

    model = Model(inputs=input_data, outputs=decoded)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model
