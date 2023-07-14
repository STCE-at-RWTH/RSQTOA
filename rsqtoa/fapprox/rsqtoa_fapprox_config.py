import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

epochs = 100
""" Amount of leaning epochs """

epsilon = 1e-5
""" Epsilon that we eant to reach """

batch_size = 256
""" Tensorflow batch size """

shuffle_buffer_size = 10000
""" Buffer size used for shuffling the training set """

learn_gradients = False
""" Whether to learn the gradients """

optimizer = Adam(learning_rate=0.00001)
""" The optimizer use for the function approximation """

loss = MeanSquaredError()
""" The loss functional uses for training """

max_y = np.empty(0)
""" The maximal values in the training data """

min_y = np.empty(0)
""" The minimal values in the training data """
