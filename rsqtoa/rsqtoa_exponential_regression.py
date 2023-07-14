import warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

# Import abstract parent class
from rsqtoa.rsqtoa_subspace_regression import QuasiSeparabilityRegression

# Import global config
from rsqtoa import rsqtoa_config as config

# ---------------------------------------------------------------------------- #

def exp_regressor(x, a, b, c):
  """ Exponential regressor """
  return a * np.exp(b * x) + c

def exp_regressor_jac(x, a, b, c):
  """ Jacobian of exponential regressor """
  jac = np.zeros((3, len(x)))
  jac[0,:] = np.exp(b * x)
  jac[1,:] = x * a * np.exp(b * x)
  jac[2,:] = 1

  return jac

# ---------------------------------------------------------------------------- #

class QuasiExponentialRegression(QuasiSeparabilityRegression):
  def __init__(self, model, dims, lower, upper, regression_dim):
    """ Constructor of the exponential subspace regression model """
    QuasiSeparabilityRegression.__init__(
      self, model, dims, lower, upper, regression_dim)
    self.coefficients = np.empty(2)

  def try_fit(self, eps):
    """ Applies the exponential regression in one dimension """

    # Quick return if we don't want exponential regressions
    if config.exponential_regression_samples == 0:
      self.max_error = np.Inf
      return False

    if config.debug:
      print(
        "Dimension " + str(self.regression_dim) +
        " exponential regression ... ", flush=True)

    # Tries to fit the current regression config
    self.fit()
    return self.validate(eps)

  def fit_at_sample(self, sample):
    """ Performs the regression over the subspace defined by the sample """
    n_samples = config.exponential_regression_samples
    sample = np.full(1, sample) if self.dims == 1 else sample

    # Sample random points in the subspace
    X, y = self.sampler.sample_dimension(n_samples, self.regression_dim, sample)
    # y = self.sampler.eval_samples()

    # Compute regression X
    X_reg = np.empty(n_samples)
    for i in range(n_samples):
      X_i = np.full(1, X[i]) if self.dims == 1 else X[i]
      X_reg[i] = X_i[self.regression_dim]

    # Lower bound, upper bound and mid point
    distance = self.upper[self.regression_dim] - self.lower[self.regression_dim]
    lb = np.copy(sample)
    lb[self.regression_dim] = self.lower[self.regression_dim]
    mid = np.copy(sample)
    mid[self.regression_dim] = distance / 2.0
    ub = np.copy(sample)
    ub[self.regression_dim] = self.upper[self.regression_dim]

    # Disable scipy warnings
    warnings.simplefilter("ignore", OptimizeWarning)
    warnings.simplefilter("ignore", RuntimeWarning)

    # Estimate for the coefficients
    coeffs = np.full(3, np.Inf)
    b = self.derivative_at(lb)
    if b == 0:
      return coeffs[:-1]
    b = self.derivative_at(ub) / b
    if b <= 0:
      return coeffs[:-1]

    b = np.log(b) / distance
    tmp = np.exp(b * mid[self.regression_dim])
    if b * tmp == 0:
      return coeffs[:-1]
    a = self.derivative_at(mid) / (b * tmp)
    c = self.model.get_target(mid) - a * tmp

    coeffs[0] = a
    coeffs[1] = b
    coeffs[2] = c

    # Perform non-linear regression
    coeffs, _ = curve_fit(exp_regressor, X_reg, y, p0=coeffs)

    # Reenable scipy warnings
    warnings.simplefilter("default", OptimizeWarning)
    warnings.simplefilter("default", RuntimeWarning)


    return coeffs[:-1]

  def __call__(self, x):
    """ Predicts the value at point x """
    return exp_regressor(x, *self.coefficients, 0)

  def __str__(self):
    """ Creates a string representation with possible line breaks """
    code = str(self.coefficients[0]) + " * std::exp($lb$"
    code += str(self.coefficients[1]) + " * x$lb$)"

    return code
