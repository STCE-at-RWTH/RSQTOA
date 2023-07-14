import sys
import os
import numpy as np
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import importlib

import pathlib
current_path = pathlib.Path(__file__).parent.resolve()
repo_path = pathlib.Path(current_path).parent.parent.parent.resolve()

sys.path.append(str(repo_path))
sys.path.append(os.path.join(str(repo_path), "build", "lib"))
import rsqtoa

model_module = importlib.import_module("diffusion_binding")
model = getattr(model_module, "diffusion")

config = rsqtoa.create_config(
  cfg=os.path.join(str(current_path), "diffusion.yml"))

ss_tree = rsqtoa.create_subspace_tree(config, model)

ss_tree.print_info()
exit()

# Train all reduced models
N_training = 10000
N_test = 2000
leafs = ss_tree.get_leafs()
coeff = 0
for leaf in leafs:
  if len(leaf.non_separable_dims) > 0:
    coeff += 1

losses = []
for leaf in leafs:
  if len(leaf.non_separable_dims) > 0:
    training_data_x = leaf.sample_domain(int(N_training / coeff))
    test_data_x = leaf.sample_domain(int(N_test / coeff))

    training_data_y = leaf.evaluate_reduced_model(training_data_x.transpose())
    training_data_y = np.expand_dims(training_data_y, axis=1)
    test_data_y = leaf.evaluate_reduced_model(test_data_x.transpose())
    test_data_y = np.expand_dims(test_data_y, axis=1)
    idx = rsqtoa.normalize_data(training_data_y, test_data_y)

    # Create reduced data set
    input_n = len(leaf.non_separable_dims)
    for i in reversed(range(leaf.dims)):
      if i not in leaf.non_separable_dims:
        training_data_x = np.delete(training_data_x, i, 1)
        test_data_x = np.delete(test_data_x, i, 1)

    # Create model
    model = rsqtoa.FunctionApproximator()
    model.add(tf.keras.Input(shape=(input_n,)))
    model.add(tf.keras.layers.Dense(100, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(10, activation='elu'))
    model.add(tf.keras.layers.Dense(1, activation='elu'))
    loss_hist = rsqtoa.train_model(
      model, training_data_x, training_data_y, test_data_x, test_data_y)
    losses.append(loss_hist)

    leaf.approximated_model = model
    leaf.denormalization_idx = idx

ss_tree.print_info()

plt.rcParams.update({'font.size': 12})

# Loss figure
fig_loss = plt.figure()

ax = fig_loss.add_subplot(1, 1, 1)
for i, loss_hist in enumerate(losses):
  ax.plot(loss_hist['train_loss_points'][0], loss_hist['train_loss_points'][1], label='Training Loss ({})'.format(i))
  ax.plot(loss_hist['val_test_loss_points'][0], loss_hist['val_test_loss_points'][1], label='Test Loss ({})'.format(i))
ax.grid(True)
ax.legend()

# Surface plots
fig = plt.figure(figsize=plt.figaspect(0.3))
fig_approx = plt.figure(figsize=plt.figaspect(0.3))

n = 21

s0 = np.linspace(50, 60, n)
K = np.linspace(52, 62, n)
r = np.linspace(0.04, 0.05, n)
sigma = np.linspace(0.01, 0.02, n)
s0_x4d, K_x4d, r_x4d, sigma_x4d = np.meshgrid(s0, K, r, sigma)
x_points = np.stack((s0_x4d, K_x4d, r_x4d, sigma_x4d))
x_points_flat = np.stack((s0_x4d.flatten(), K_x4d.flatten(), r_x4d.flatten(), sigma_x4d.flatten()))
Z_full = black_scholes(x_points).squeeze()
Z_approx_full = ss_tree.evaluate_reassembled_model(x_points_flat).reshape(np.shape(Z_full))
error = np.abs(Z_full - Z_approx_full)

# s0 = np.linspace(50, 70, 101)
# K = np.linspace(50, 70, 101)
# r = np.linspace(0.01, 0.1, 101)
# sigma = np.linspace(0.01, 0.1, 101)

anchor_s0 = 53
anchor_K = 58
anchor_r = 0.045
anchor_sigma = 0.015

# S0 & K
s0_x, K_x, r_x, sigma_x = np.meshgrid(s0, K, anchor_r, anchor_sigma)
x_points = np.stack((s0_x, K_x, r_x, sigma_x))
Z = black_scholes(x_points).squeeze()

X, Y = np.meshgrid(s0, K)

ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.plot_surface(X, Y, Z, shade=True)
ax.set_xlabel("$S_0$", labelpad=10, fontsize=20)
ax.set_ylabel("$K$", labelpad=10, fontsize=20)
ax.set_title("$r = {0}$ and $\sigma = {1}$".format(anchor_r, anchor_sigma))

ax = fig_approx.add_subplot(2, 3, 1, projection='3d')
ax.plot_surface(X, Y, np.average(error, axis=(2,3)), shade=True)
ax.set_xlabel("$S_0$", labelpad=10, fontsize=20)
ax.set_ylabel("$K$", labelpad=10, fontsize=20)
ax.set_title("1-norm loss surface averaged over $r$ and $\sigma$")

# S0 & r
s0_x, K_x, r_x, sigma_x = np.meshgrid(s0, anchor_K, r, anchor_sigma)
x_points = np.stack((s0_x, K_x, r_x, sigma_x))
x_points_flat = np.stack((s0_x.flatten(), K_x.flatten(), r_x.flatten(), sigma_x.flatten()))
Z = black_scholes(x_points).squeeze()
Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)
Z_approx = Z_approx.reshape(np.shape(Z))

X, Y = np.meshgrid(s0, r)

ax = fig.add_subplot(2, 3, 2, projection='3d')
ax.plot_surface(X, Y, Z, shade=True)
ax.set_xlabel("$S_0$", labelpad=10, fontsize=20)
ax.set_ylabel("$r$", labelpad=10, fontsize=20)
ax.set_title("$K = {0}$ and $\sigma = {1}$".format(anchor_K, anchor_sigma))

ax = fig_approx.add_subplot(2, 3, 2, projection='3d')
ax.plot_surface(X, Y, np.average(error, axis=(1,3)), shade=True)
ax.set_xlabel("$S_0$", labelpad=10, fontsize=20)
ax.set_ylabel("$r$", labelpad=10, fontsize=20)
ax.set_title("1-norm loss surface averaged over $K$ and $\sigma$")

# S0 & sigma
s0_x, K_x, r_x, sigma_x = np.meshgrid(s0, anchor_K, anchor_r, sigma)
x_points = np.stack((s0_x, K_x, r_x, sigma_x))
x_points_flat = np.stack((s0_x.flatten(), K_x.flatten(), r_x.flatten(), sigma_x.flatten()))
Z = black_scholes(x_points).squeeze()
Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)
Z_approx = Z_approx.reshape(np.shape(Z))

X, Y = np.meshgrid(s0, sigma)

ax = fig.add_subplot(2, 3, 3, projection='3d')
ax.plot_surface(X, Y, Z, shade=True)
ax.set_xlabel("$S_0$", labelpad=10, fontsize=20)
ax.set_ylabel("$\sigma$", labelpad=10, fontsize=20)
ax.set_title("$K = {0}$ and $r = {1}$".format(anchor_K, anchor_r))

ax = fig_approx.add_subplot(2, 3, 3, projection='3d')
ax.plot_surface(X, Y, np.average(error, axis=(1,2)), shade=True)
ax.set_xlabel("$S_0$", labelpad=10, fontsize=20)
ax.set_ylabel("$\sigma$", labelpad=10, fontsize=20)
ax.set_title("1-norm loss surface averaged over $K$ and $r$")

# K & r
s0_x, K_x, r_x, sigma_x = np.meshgrid(anchor_s0, K, r, anchor_sigma)
x_points = np.stack((s0_x, K_x, r_x, sigma_x))
x_points_flat = np.stack((s0_x.flatten(), K_x.flatten(), r_x.flatten(), sigma_x.flatten()))
Z = black_scholes(x_points).squeeze()
Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)
Z_approx = Z_approx.reshape(np.shape(Z))

X, Y = np.meshgrid(K, r)

ax = fig.add_subplot(2, 3, 4, projection='3d')
ax.plot_surface(X, Y, Z, shade=True)
ax.set_xlabel("$K$", labelpad=10, fontsize=20)
ax.set_ylabel("$r$", labelpad=10, fontsize=20)
ax.set_title("$S_0 = {0}$ and $\sigma = {1}$".format(anchor_s0, anchor_sigma))

ax = fig_approx.add_subplot(2, 3, 4, projection='3d')
ax.plot_surface(X, Y, np.average(error, axis=(0,3)), shade=True)
ax.set_xlabel("$K$", labelpad=10, fontsize=20)
ax.set_ylabel("$r$", labelpad=10, fontsize=20)
ax.set_title("1-norm loss surface averaged over $S_0$ and $\sigma$")

# K & sigma
s0_x, K_x, r_x, sigma_x = np.meshgrid(anchor_s0, K, anchor_r, sigma)
x_points = np.stack((s0_x, K_x, r_x, sigma_x))
x_points_flat = np.stack((s0_x.flatten(), K_x.flatten(), r_x.flatten(), sigma_x.flatten()))
Z = black_scholes(x_points).squeeze()
Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)
Z_approx = Z_approx.reshape(np.shape(Z))

X, Y = np.meshgrid(K, sigma)

ax = fig.add_subplot(2, 3, 5, projection='3d')
ax.plot_surface(X, Y, Z, shade=True)
ax.set_xlabel("$K$", labelpad=10, fontsize=20)
ax.set_ylabel("$\sigma$", labelpad=10, fontsize=20)
ax.set_title("$S_0 = {0}$ and $r = {1}$".format(anchor_s0, anchor_r))

ax = fig_approx.add_subplot(2, 3, 5, projection='3d')
ax.plot_surface(X, Y, np.average(error, axis=(0,2)), shade=True)
ax.set_xlabel("$K$", labelpad=10, fontsize=20)
ax.set_ylabel("$\sigma$", labelpad=10, fontsize=20)
ax.set_title("1-norm loss surface averaged over $S_0$ and $r$")

# r & sigma
s0_x, K_x, r_x, sigma_x = np.meshgrid(anchor_s0, anchor_K, r, sigma)
x_points = np.stack((s0_x, K_x, r_x, sigma_x))
x_points_flat = np.stack((s0_x.flatten(), K_x.flatten(), r_x.flatten(), sigma_x.flatten()))
Z = black_scholes(x_points).squeeze()
Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)
Z_approx = Z_approx.reshape(np.shape(Z))

X, Y = np.meshgrid(r, sigma)

ax = fig.add_subplot(2, 3, 6, projection='3d')
ax.plot_surface(X, Y, Z, shade=True)
ax.set_xlabel("$r$", labelpad=10, fontsize=20)
ax.set_ylabel("$\sigma$", labelpad=10, fontsize=20)
ax.set_title("$S_0 = {0}$ and $K = {1}$".format(anchor_s0, anchor_K))
ax.tick_params(axis='z', pad=15)

ax = fig_approx.add_subplot(2, 3, 6, projection='3d')
ax.plot_surface(X, Y, np.average(error, axis=(0,1)), shade=True)
ax.set_xlabel("$r$", labelpad=10, fontsize=20)
ax.set_ylabel("$\sigma$", labelpad=10, fontsize=20)
ax.set_title("1-norm loss surface averaged over $S_0$ and $K$")
ax.tick_params(axis='z', pad=15)

plt.show()

exit(0)

X_validate = ss_tree.root.sample_domain(10000)
Z = black_scholes(X_validate.transpose())
Z_approx = ss_tree.evaluate_reassembled_model(X_validate.transpose())

diff = (Z - Z_approx)
print(np.max(np.abs(diff)))

exit(0)

X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(X, Y)
x_points = np.stack((X, Y))
x_points_flat = np.stack((X.flatten(), Y.flatten()))

fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})

Z = f(x_points)
axs[0, 0].plot_surface(X, Y, Z, shade=True)
axs[0, 1].plot_surface(X, Y, ss_tree.evaluate_reduced_model(x_points))

Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)
Z_approx = Z_approx.reshape(np.shape(Z))
axs[1, 0].plot_surface(X, Y, Z_approx)
axs[1, 1].plot_surface(X, Y, np.average(error, axis=(2,3)))

plt.show()
