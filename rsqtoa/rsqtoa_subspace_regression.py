import sys
import numpy as np

# Import abstract parent class
from abc import ABC, abstractmethod, abstractclassmethod

# Import global config
from rsqtoa import rsqtoa_config as config
from rsqtoa.rsqtoa_sampling import RandomSampler, LatinHypercubeSampler, GridSampler
from rsqtoa.rsqtoa_fd import _fd_derivative

# ---------------------------------------------------------------------------- #

class QuasiSeparabilityRegression(ABC):
  def __init__(self, model, dims, lower, upper, regression_dim):
    """ Constructor of the quasi-separability regression base class  """
    self.model = model
    self.dims = dims
    self.lower = lower.astype(float)
    self.upper = upper.astype(float)
    self.regression_dim = regression_dim
    self.coefficients = None
    self.max_error = 0.0
    self.sample_value = 0.0

    if config.sampling_strategy == 'random':
      self.sampler = RandomSampler(model, dims, lower, upper)
    elif config.sampling_strategy == 'dataset':
      self.sampler = GridSampler(model, dims, lower, upper)
    else:
      self.sampler = LatinHypercubeSampler(model, dims, lower, upper)

  def derivative_at(self, point):
    """ Return the partial derivative at a given sample """
    return _fd_derivative(
      self.model, self.lower, self.upper, self.regression_dim, point)

  def fit(self):
    """ Tries to fit the regression model over a subspace of the domain """

    # Get random samples in the entire domain
    X, y = self.sampler.sample_domain(config.regression_samples, self.regression_dim)

    # Perform subspace regression at samples and take average of coefficients
    try:
      self.coefficients = np.zeros_like(self.coefficients)
      for X_sample in X:
        self.coefficients += self.fit_at_sample(X_sample)
      self.coefficients /= config.regression_samples
    except KeyboardInterrupt:
      sys.exit()
    # except Exception as e:
    #   print(e)
    #   self.max_error = np.Inf
    #   return

    # Quick return in case of infinite or NaN coefficeints
    if any(np.isnan(self.coefficients)) or any(np.isinf(self.coefficients)):
      self.max_error = np.Inf
      return

    # Get random test points in the entire domain (N=config.test_samples)
    X_test, y_test = self.sampler.sample_domain(config.test_samples, self.regression_dim)
    # y_test = self.sampler.eval_samples()

    # Errors
    err_matrix = np.zeros((config.test_samples, config.test_values))

    for i in range(config.test_samples):
      # Evaluate regression model with x_i from the test point
      y_test_reg = self(X_test[i][self.regression_dim])

      # For each test point, sample a certain number of points in dimension x_i
      # with the test point as anchor
      X_tilde, y_tilde = self.sampler.sample_dimension(
        config.test_values, self.regression_dim, X_test[i])
      # y_tilde = self.sampler.eval_samples()

      for j in range(config.test_values):
        # Evaluate regression model with the additionally sampled x_i
        y_tilde_reg = self(X_tilde[j][self.regression_dim])

        # Calculate error (see Quasi-Separability definition)
        err_matrix[i,j] = abs(y_test_reg + y_tilde[j] - y_tilde_reg - y_test[i])

    # Caluclate maximal error and choose a sample value for the model reduction
    self.max_error = err_matrix.max()
    self.sample_value = (
      X_tilde[err_matrix.max(0).argmin()][self.regression_dim])

  def validate(self, current_epsilon):
    """ Validated the maximal error against the given error threshold """
    success = self.max_error < current_epsilon
    if success:
      if config.debug:
        print(
          "success: max error (" + str(self.max_error) + ") < epsilon ("
          + str(current_epsilon) + ")")
    else:
      if config.debug:
        print(
          "fail: max error (" + str(self.max_error) + ") > epsilon ("
          + str(current_epsilon) + ")")

    return success

  @abstractclassmethod
  def try_fit(self, eps):
    pass

  @abstractclassmethod
  def fit_at_sample(self, sample):
    pass

  @abstractmethod
  def __call__(self, x):
    pass

  @abstractmethod
  def __str__(self):
    pass
