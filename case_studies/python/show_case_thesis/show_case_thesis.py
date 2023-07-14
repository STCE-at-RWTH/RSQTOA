import sys
import os
from matplotlib import projections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pathlib
current_path = pathlib.Path(__file__).parent.resolve()
repo_path = pathlib.Path(current_path).parent.parent.parent.resolve()

sys.path.append(str(repo_path))
import rsqtoa
from rsqtoa.model_wrapper import EquationModel

# The original model
def f(x):
  tmp = 2 + x[0] * np.sin(5*x[0])
  if np.isscalar(x[1]):
    if x[1] > 0.3:
      tmp += 10*(x[1]-0.3) ** 2
  else:
    tmp[x[1] > 0.3] += 10*(x[1][x[1] > 0.3]-0.3) ** 2

  return tmp

# Equidistant samples for plotting
X = np.arange(0, 1, 0.001)
Y = np.arange(0, 1, 0.001)
X, Y = np.meshgrid(X, Y)
x_points = np.stack((X, Y))
x_points_flat = np.stack((X.flatten(), Y.flatten()))
Z = f(x_points)

plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
axs.plot_surface(X, Y, Z)
plt.show()
# exit(0)

# Create / load config
config = rsqtoa.create_config(
  cfg=os.path.join(str(current_path), "show_case_thesis.yml"))

model = EquationModel(f, config['dimensions'], config['lower_bounds'], config['upper_bounds'])

# Run RSQTOA framework which creates the subspace tree
ss_tree = rsqtoa.create_subspace_tree(config, model)

plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
axs.tick_params(axis='z', pad=15)
Z = ss_tree.evaluate_reduced_model(x_points)
axs.plot_surface(X, Y, Z)
plt.show()
# exit(0)

# plt.rcParams.update({'font.size': 20})
# fig, axs = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
# axs.tick_params(axis='z', pad=25)
# Z_approx = ss_tree.evaluate_reassembled_model(x_points)
# axs.plot_surface(X, Y, np.abs(Z - Z_approx))
# plt.show()
# exit(0)

# Train all reduced models
N_training = 1000
N_test = 200
leafs = ss_tree.get_leafs() # Returns all subspaces
coeff = 0
for leaf in leafs:
  if len(leaf.non_separable_dims) > 0:
    coeff += 1

losses = []
for leaf in leafs:
  if len(leaf.non_separable_dims) > 0:
    # Create samples in the subspace (training & test data)
    training_data_x, training_data_y = leaf.sample_domain(int(N_training / coeff))
    test_data_x, test_data_y = leaf.sample_domain(int(N_test / coeff))

    # Evaluate the reduced model at the training samples
    training_data_y = leaf.evaluate_reduced_model(training_data_x.transpose())
    training_data_y = np.expand_dims(training_data_y, axis=1)

    # Evaluate the reduced model at the test samples
    test_data_y = leaf.evaluate_reduced_model(test_data_x.transpose())
    test_data_y = np.expand_dims(test_data_y, axis=1)

    # Normalize the data for training
    idx = rsqtoa.normalize_data(training_data_y, test_data_y)

    # Create reduced data set
    input_n = len(leaf.non_separable_dims)
    for i in reversed(range(leaf.dims)):
      if i not in leaf.non_separable_dims:
        training_data_x = np.delete(training_data_x, i, 1)
        test_data_x = np.delete(test_data_x, i, 1)

    # Create tensorflow model
    model = rsqtoa.FunctionApproximator()
    model.add(tf.keras.Input(shape=(input_n,)))
    model.add(tf.keras.layers.Dense(100, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(10, activation='elu'))
    model.add(tf.keras.layers.Dense(1, activation='elu'))

    # Train the tensorflow model
    loss_hist = rsqtoa.train_model(
      model, training_data_x, training_data_y, test_data_x, test_data_y)
    losses.append(loss_hist)

    # Set the trained tensorflow model in the subspace node
    leaf.approximated_model = model

    # Set the nomalization index so taht we can denormalize the results
    leaf.denormalization_idx = idx

# Print some debug info
ss_tree.print_info()

# Plot results
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1)
for i, loss_hist in enumerate(losses):
  ax.plot(loss_hist['train_loss_points'][0], loss_hist['train_loss_points'][1], label='Training Loss ({})'.format(i))
  ax.plot(loss_hist['val_test_loss_points'][0], loss_hist['val_test_loss_points'][1], label='Test Loss ({})'.format(i))
ax.grid(True)
ax.legend()
plt.show()

# ax = fig.add_subplot(1, 1, 1, projection='3d')
# Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)
# Z_approx = Z_approx.reshape(np.shape(Z))
# ax.tick_params(axis='z', pad=15)
# ax.plot_surface(X, Y, np.abs(Z - Z_approx))
# plt.show()
# exit(0)

fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})

Z = f(x_points)
axs[0, 0].plot_surface(X, Y, Z)
axs[0, 1].plot_surface(X, Y, ss_tree.evaluate_reduced_model(x_points))

Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)
Z_approx = Z_approx.reshape(np.shape(Z))
axs[1, 0].plot_surface(X, Y, Z_approx)
axs[1, 1].plot_surface(X, Y, np.abs(Z - Z_approx))

plt.show()
