import tensorflow as tf

import rsqtoa


def get_ann_function_approximator(input_dimension: int) -> rsqtoa.FunctionApproximator:
    model = rsqtoa.FunctionApproximator()

    model.add(tf.keras.layers.Dense(50, input_dim=input_dimension, activation='elu'))
    model.add(tf.keras.layers.Dense(100, activation='elu'))
    model.add(tf.keras.layers.Dense(100, activation='elu'))
    model.add(tf.keras.layers.Dense(50, activation='elu'))
    model.add(tf.keras.layers.Dense(10, activation='elu'))
    model.add(tf.keras.layers.Dense(1, activation='elu'))

    model.summary()

    return model

