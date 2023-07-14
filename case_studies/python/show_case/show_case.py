import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pathlib
current_path = pathlib.Path(__file__).parent.resolve()
repo_path = pathlib.Path(current_path).parent.parent.parent.resolve()

sys.path.append(str(repo_path))
import rsqtoa


def f(x):
  tmp = 4.0 * np.abs(x[0] - 0.3) + 2.0
  if np.isscalar(x[0]):
    if x[0] > 0.125:
      tmp += 7.6 * (x[1] - 0.5) * (x[1] - 0.5)
    else:
      tmp += 10 * np.sqrt(x[1]+0.1)
  else:
    right = x[0] > 0.125
    left = x[0] <= 0.125
    tmp[right] += 7.6 * (x[1][right] - 0.5) * (x[1][right] - 0.5)
    tmp[left] += 10 * np.sqrt(x[1][left] + 0.1)

  return tmp

config = rsqtoa.create_config(
  cfg=os.path.join(str(current_path), "show_case.yml"))

ss_tree = rsqtoa.create_subspace_tree(config, f)

# Train all reduced models
N_training = 10000
N_test = 1000
leafs = ss_tree.get_leafs()
for leaf in leafs:
  if len(leaf.non_separable_dims) > 0:
    coeff = leaf.get_sample_coefficient()
    training_data_x = leaf.sample_domain(int(N_training))
    test_data_x = leaf.sample_domain(int(N_test))

    training_data_y = leaf.evaluate_reduced_model(training_data_x.transpose())
    training_data_y = np.expand_dims(training_data_y, axis=1)
    test_data_y = leaf.evaluate_reduced_model(test_data_x.transpose())
    test_data_y = np.expand_dims(test_data_y, axis=1)
    idx = rsqtoa.normalize_data(training_data_y, test_data_y)

    # Create model
    model = rsqtoa.FunctionApproximator()
    model.add(tf.keras.Input(shape=(2,)))
    model.add(tf.keras.layers.Dense(100, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(10, activation='elu'))
    model.add(tf.keras.layers.Dense(1, activation='elu'))
    rsqtoa.train_model(
      model, training_data_x, training_data_y, test_data_x, test_data_y)

    leaf.approximated_model = lambda x: model.predict(x.transpose()).squeeze()
    leaf.denormalization_idx = idx


X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(X, Y)
x_points = np.stack((X, Y))
x_points_flat = np.stack((X.flatten(), Y.flatten()))

fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})

Z = f(x_points)
axs[0, 0].plot_surface(X, Y, Z)
axs[0, 1].plot_surface(X, Y, ss_tree.evaluate_reduced_model(x_points))

Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)
Z_approx = Z_approx.reshape(np.shape(Z))
axs[1, 0].plot_surface(X, Y, Z_approx)
axs[1, 1].plot_surface(X, Y, np.abs(Z - Z_approx))

plt.show()
