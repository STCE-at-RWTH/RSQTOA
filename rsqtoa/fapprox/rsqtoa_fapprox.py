import numpy as np
import rsqtoa.fapprox.rsqtoa_fapprox_config as config
from rsqtoa.fapprox.rsqtoa_trainig_data import _build_dataset
from rsqtoa.fapprox.rsqtoa_tf_models import DeepFunctionApproximator

def train_model(model, train_data_x, train_data_y, test_data_x, test_data_y):
  """ train_dataset and test_dataset assumed to be (xs, ys, ys_dx), or (xs, ys) """
  return model.compile_and_fit(
    _build_dataset(config.batch_size, train_data_x, train_data_y),
    _build_dataset(config.batch_size, test_data_x, test_data_y))

def normalize_data(*data_y):
  """ Normalize the y values to a value between -1 and 1 """
  idx = len(config.max_y)
  config.max_y = np.append(config.max_y, -np.Inf)
  config.min_y = np.append(config.min_y, np.Inf)

  for y in data_y:
    config.max_y[idx] = np.max([config.max_y[idx], y.max()])
    config.min_y[idx] = np.min([config.min_y[idx], y.min()])

  for y in data_y:
    y[:] = (y[:] - config.min_y[idx]) / (config.max_y[idx] - config.min_y[idx])

  return idx

def denormalize_data(y, idx):
  """ Denormalize the y values """
  print("idx = {}".format(idx))
  y[:] = config.min_y[idx] + y[:] * (config.max_y[idx] - config.min_y[idx])
  return y

def reset_normalization():
  """ Resets the normalization coefficients """
  config.max_y = np.empty(0)
  config.min_y = np.empty(0)
