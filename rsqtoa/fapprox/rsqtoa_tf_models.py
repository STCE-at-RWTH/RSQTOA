import tensorflow as tf
from rsqtoa.fapprox.rsqtoa_tf_function_approximator import FunctionApproximator

# ***************************************************
# Backbones for the function-learning model
# ***************************************************

class TinyFunctionApproximator(FunctionApproximator):
  """ Small network """
  def __init__(self):
    super(TinyFunctionApproximator, self).__init__()

    self.d1 = tf.keras.layers.Dense(100, activation='relu')
    self.d2 = tf.keras.layers.Dense(100, activation='relu')
    self.d3 = tf.keras.layers.Dense(10, activation='relu')
    self.d4 = tf.keras.layers.Dense(1, activation='relu')
    self.d5 = tf.keras.layers.Dense(1, activation='relu')

  def call(self, x):
    x = self.d5(self.d4(self.d3(self.d2(self.d1(x)))))
    return x

class MediumFunctionApproximator(FunctionApproximator):
  """ Medium sized network """
  def __init__(self):
    super(MediumFunctionApproximator, self).__init__()

    self.d1 = tf.keras.layers.Dense(100, activation='relu')
    self.d2 = tf.keras.layers.Dense(1000, activation='relu')
    self.d3 = tf.keras.layers.Dense(1000, activation='relu')
    self.d4 = tf.keras.layers.Dense(10, activation='relu')
    self.d5 = tf.keras.layers.Dense(1, activation='relu')

  def call(self, x):
    x = self.d5(self.d4(self.d3(self.d2(self.d1(x)))))
    return x

class DeepFunctionApproximator(FunctionApproximator):
  """ Deep network """
  def __init__(self):
    super(DeepFunctionApproximator, self).__init__()

    self.d1 = tf.keras.layers.Dense(
      100, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
    self.d2 = tf.keras.layers.Dense(1000, activation='relu')
    self.d3 = tf.keras.layers.Dense(1000, activation='relu')
    self.d4 = tf.keras.layers.Dense(1000, activation='relu')
    self.d5 = tf.keras.layers.Dense(1000, activation='relu')
    self.d6 = tf.keras.layers.Dense(10, activation='relu')
    self.d7 = tf.keras.layers.Dense(1, activation='relu')

  def call(self, x):
    q = self.d3(self.d2(self.d1(x)))
    x = self.d7(self.d6(self.d5(self.d4(q))))
    return x
