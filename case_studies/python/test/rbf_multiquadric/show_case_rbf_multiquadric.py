import sys
import os
import random
from matplotlib import projections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.interpolate import RBFInterpolator
import pathlib
from timeit import default_timer as timer

current_path = pathlib.Path(__file__).parent.resolve()
repo_path = pathlib.Path(current_path).parent.parent.parent.resolve()

sys.path.append(str(repo_path))
import rsqtoa


df_train = pd.read_csv('./../../train_sample_10000.csv', index_col=[0])

X = df_train.iloc[:, 0].values
Y = df_train.iloc[:, 1].values
Z = df_train.iloc[:, 2].values

x_points_flat = np.stack((X, Y))

start = timer()
ip = RBFInterpolator(x_points_flat.T, Z, kernel='multiquadric', epsilon=6)

Z_ip = ip(x_points_flat.T)


def f_interpol(x):
    z_ip = float(ip([x])) if np.shape(x) == (2,) else ip(x.T)
    return z_ip


# Create / load config
config = rsqtoa.create_config(
    cfg=os.path.join(str(current_path), "show_case.yml"))
print('Total compile time ::', timer() - start)

start = timer()
# Run RSQTOA framework which creates the subspace tree
ss_tree = rsqtoa.create_subspace_tree(config, f_interpol)
ss_tree.print_info()

print('Model Reduction time ::', timer() - start)

# exit(0)

start = timer()

# Train all reduced models
N_training = 1000
N_test = 200
leafs = ss_tree.get_leafs()  # Returns all subspaces
coeff = 0
for leaf in leafs:
    if len(leaf.non_separable_dims) > 0:
        coeff += 1

losses = []
for leaf in leafs:
    if len(leaf.non_separable_dims) > 0:
        # Create samples in the subspace (training & test data)
        training_data_x = leaf.sample_domain(int(N_training / coeff))
        test_data_x = leaf.sample_domain(int(N_test / coeff))

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
print('Residual Domain model training time ::', timer() - start)

# Print some debug info
ss_tree.print_info()


Z_approx = ss_tree.evaluate_reassembled_model(x_points_flat)

df = pd.DataFrame(
    np.stack([X.flatten(), Y.flatten(), Z.flatten(), Z_approx.flatten(), np.abs(Z - Z_approx).flatten()], axis=1),
    columns=['x', 'y', 'z', 'z_approx', 'abs_error']
)
df.to_csv('train_results.csv')

print('Mean Squared Error (Training Data):: ', mean_squared_error(Z.flatten(), Z_approx.flatten()))
print('Root Mean Squared Error (Training Data):: ', mean_squared_error(Z.flatten(), Z_approx.flatten(), squared=False))

df_test = pd.read_csv('./../../test_sample_1000.csv')
test_array = np.stack((df_test['x'].values, df_test['y'].values))

start = timer()
res_test_model = ss_tree.evaluate_reassembled_model(test_array)
print('Total prediction time ::', timer() - start)

res_test_original = df_test['z'].values

df_test['z_approx'] = res_test_model
df_test['abs_error'] = np.abs(res_test_original - res_test_model)
df_test.to_csv('test_result.csv')

print('Mean Squared Error (Out of sample test):: ', mean_squared_error(res_test_original, res_test_model))
print('Root Mean Squared Error (Out of sample test):: ', mean_squared_error(res_test_original, res_test_model, squared=False))
