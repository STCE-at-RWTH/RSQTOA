import sys
import os
import numpy as np
import tensorflow as tf
from scipy import integrate
import matplotlib.pyplot as plt

import pathlib
current_path = pathlib.Path(__file__).parent.resolve()
repo_path = pathlib.Path(current_path).parent.parent.parent.resolve()

sys.path.append(str(repo_path))
import rsqtoa

plt.rcParams.update({'font.size': 12})

# ODE
def f_plot(t, a, b, c, d, X0):
  def dX_dt(X, t=0):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([a*X[0] - b*X[0]*X[1], d*b*X[0]*X[1] - c*X[1]])

  return integrate.odeint(dX_dt, X0, t)

# ODE
def f(x):
  x = x.squeeze()
  t = np.linspace(0, x[0], 1000)
  a = x[1]
  b = x[2]
  c = x[3]
  d = x[4]
  X0 = x[5:]

  def dX_dt(X, t=0):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([a*X[0] - b*X[0]*X[1], d*b*X[0]*X[1] - c*X[1]])

  X = integrate.odeint(dX_dt, X0, t)
  return X[-1][0]

config = rsqtoa.create_config(
  cfg=os.path.join(str(current_path), "lotka_volterra.yml"))

ss_tree = rsqtoa.create_subspace_tree(config, f)

# Plot figure
fig_plot = plt.figure()
t = np.linspace(0, 5, 10000)
X = f_plot(t=t, a=1, b=0.1, c=1.5, d=0.75, X0=[10, 5])
ax = fig_plot.add_subplot(1, 1, 1)
ax.plot(t, X[:,0], label="Amount of prey")
ax.plot(t, X[:,1], label="Amount of predators")
ax.set_xlabel("Time $t$", labelpad=10, fontsize=20)
ax.set_ylabel("Population", labelpad=10, fontsize=20)
ax.legend()
ax.grid(True)

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
    model.add(tf.keras.layers.Dense(10000, activation='elu'))
    model.add(tf.keras.layers.Dense(10000, activation='elu'))
    model.add(tf.keras.layers.Dense(1000, activation='elu'))
    model.add(tf.keras.layers.Dense(10, activation='elu'))
    model.add(tf.keras.layers.Dense(1, activation='elu'))
    loss_hist = rsqtoa.train_model(
      model, training_data_x, training_data_y, test_data_x, test_data_y)
    losses.append(loss_hist)

    leaf.approximated_model = model
    leaf.denormalization_idx = idx

plt.rcParams.update({'font.size': 12})



# Loss figure
fig_loss = plt.figure()

ax = fig_loss.add_subplot(1, 1, 1)
for i, loss_hist in enumerate(losses):
  ax.plot(loss_hist['train_loss_points'][0], loss_hist['train_loss_points'][1], label='Training Loss ({})'.format(i))
  ax.plot(loss_hist['val_test_loss_points'][0], loss_hist['val_test_loss_points'][1], label='Test Loss ({})'.format(i))
ax.grid(True)
ax.legend()



# Loss curve
fig_curve = plt.figure()
t = np.linspace(0, 5, 10000)
tv, a, b, c, d, X0_1, X0_2 = np.meshgrid(t, 1, 0.1, 1.5, 0.75, 10, 5)
x_points = np.stack((tv.flatten(), a.flatten(), b.flatten(), c.flatten(), d.flatten(), X0_1.flatten(), X0_2.flatten()))
X_approx = ss_tree.evaluate_reassembled_model(x_points)
ax = fig_curve.add_subplot(1, 2, 1)
ax.plot(t, X[:,0], label="Amount of prey (ODE)")
ax.plot(t, X_approx, label="Amount of prey (Approximation)")
ax.set_xlabel("Time $t$", labelpad=10, fontsize=20)
ax.set_ylabel("Population", labelpad=10, fontsize=20)
ax.legend()
ax.grid(True)

ax = fig_curve.add_subplot(1, 2, 2)
ax.plot(t, np.abs(X_approx - X[:,0]), label="Amount of prey (Loss)")
ax.set_xlabel("Time $t$", labelpad=10, fontsize=20)
ax.set_ylabel("Population (Loss)", labelpad=10, fontsize=20)
ax.legend()
ax.grid(True)


plt.show()





exit(0)

X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(X, Y)
x_points = np.stack((X, Y))

fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})

Z = f(x_points)
axs[0, 0].plot_surface(X, Y, Z)
axs[0, 1].plot_surface(X, Y, ss_tree.evaluate_reduced_model(x_points))

Z_approx = ss_tree.evaluate_reassembled_model(x_points)
axs[1, 0].plot_surface(X, Y, Z_approx)
axs[1, 1].plot_surface(X, Y, np.abs(Z - Z_approx))

plt.show()
