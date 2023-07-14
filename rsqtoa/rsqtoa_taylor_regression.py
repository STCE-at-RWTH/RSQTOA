import numpy as np

# Import abstract parent class
from rsqtoa.rsqtoa_subspace_regression import QuasiSeparabilityRegression

# Import global config
from rsqtoa import rsqtoa_config as config

# ---------------------------------------------------------------------------- #

class QuasiTaylorRegression(QuasiSeparabilityRegression):
  def __init__(self, model, dims, lower, upper, regression_dim):
    """ Constructor of the Taylor subspace regression model """
    QuasiSeparabilityRegression.__init__(
      self, model, dims, lower, upper, regression_dim)
    self.order = 1
    self.coefficients = np.empty(self.order)

  def try_fit(self, eps):
    """ Applies the taylor regression in one dimension """

    # Try up to the maximal taylor order. Return immediately on success. 
    for p in range(1, config.max_taylor_order + 1):
      self.order = p
      self.coefficients = np.empty(self.order)

      if config.debug:
        print(
          "Dimension " + str(self.regression_dim) + " taylor (order = " + str(p)
          + ") regression ... ", flush=True)

      # Tries to fit the current regression config
      self.fit()
      if self.validate(eps):
        return True

    return False

  def fit_at_sample(self, sample):
    """ Performs the regression over the subspace defined by the sample """

    # Number of samples is at least as large as the number of coefficents
    n_coefficients = self.order + 1
    n_samples = n_coefficients + config.taylor_regression_samples
    sample = np.full(1, sample) if self.dims == 1 else sample

    # Sample random points in the subspace
    X, y = self.sampler.sample_dimension(n_samples, self.regression_dim, sample)

    # Compute regression matrix X (linear system)
    X_reg = np.empty((n_samples, n_coefficients))
    for i in range(n_samples):
      X_i = np.full(1, X[i]) if self.dims == 1 else X[i]
      for j in range(n_coefficients):
        X_reg[i][j] = X_i[self.regression_dim] ** j

    # Perform regression via normal equation
    coeffs = np.linalg.solve(
      np.matmul(X_reg.transpose(), X_reg),
      np.matmul(X_reg.transpose(), y))

    # Return all coefficients apart from the constant term
    return coeffs[1:]

  def __call__(self, x):
    """ Predicts the value at point x """
    y = 0
    for i in range(len(self.coefficients)):
      y += self.coefficients[i] * (x ** (i+1))

    return y

  def __str__(self):
    """ Creates a string representation with possible line breaks """
    code = ""
    for i in range(len(self.coefficients)):
      code += str(self.coefficients[i])
      if i==0:
        code += " * x"
      else:
        code += " * std::pow(x, " + str(i+1) + ")"
      code += " +$lb$"

    return code[:-6]
