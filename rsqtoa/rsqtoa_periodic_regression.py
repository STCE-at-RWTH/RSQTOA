import warnings
import numpy as np
from scipy.optimize import curve_fit, fmin, OptimizeWarning

# Import abstract parent class
from rsqtoa.rsqtoa_subspace_regression import QuasiSeparabilityRegression

# Import global config
from rsqtoa import rsqtoa_config as config

# ---------------------------------------------------------------------------- #

def sin_regressor(x, a, b, c):
  """ Periodic regressor """
  return a * np.sin(b * x + c)

# ---------------------------------------------------------------------------- #

class QuasiPeriodicRegression(QuasiSeparabilityRegression):
  def __init__(self, model, dims, lower, upper, regression_dim):
    """ Constructor of the periodic subspace regression model """
    QuasiSeparabilityRegression.__init__(
      self, model, dims, lower, upper, regression_dim)
    self.coefficients = np.empty(3)

  def try_fit(self, eps):
    """ Applies the periodic regression in one dimension """

    # Quick return if we don't want periodic regressions
    if config.periodic_regression_samples == 0:
      self.max_error = np.Inf
      return False

    if config.debug:
      print(
        "Dimension " + str(self.regression_dim) +
        " periodic regression ... ", flush=True)

    # Tries to fit the current regression config
    self.fit()
    return self.validate(eps)

  def _model_slice(self, x, sample):
    """ Scalar model slice in the regression dimension. """
    x_eval = sample.copy()
    x_eval[self.regression_dim] = x
    return self.model.get_target(x_eval)

  def fit_at_sample(self, sample):
    """ Performs the regression over the subspace defined by the sample """
    n_samples = config.periodic_regression_samples
    sample = np.full(1, sample) if self.dims == 1 else sample

    # Sample random points in the subspace
    X, y = self.sampler.sample_dimension(n_samples, self.regression_dim, sample)
    # y = self.sampler.eval_samples()

    # Compute regression X
    X_reg = np.empty(n_samples)
    for i in range(n_samples):
      X_i = np.full(1, X[i]) if self.dims == 1 else X[i]
      X_reg[i] = X_i[self.regression_dim]

    # Midpoint of the scalar domain
    mid_point = self.lower[self.regression_dim]
    mid_point += (
      self.upper[self.regression_dim] - self.lower[self.regression_dim]
    ) / 2.0

    # Disable scipy warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", OptimizeWarning)

    # Estimates for the amplitude, frequency and phase
    min_x = fmin(lambda x: self._model_slice(x, sample), mid_point, disp=False)
    max_x = fmin(lambda x: -self._model_slice(x, sample), mid_point, disp=False)
    min_y = self._model_slice(min_x, sample)
    max_y = self._model_slice(max_x, sample)

    # Get the amplitude
    amp = (max_y - min_y) / 2.0

    # Get the frequency
    freq = np.pi / np.abs(min_x - max_x)

    # Get the phase
    phase = np.fmod(max_x - np.pi / (2*freq), (2*np.pi / freq))
    if phase > np.pi / freq:
      phase -= 2*np.pi / freq
    elif phase < -np.pi / freq:
      phase += 2*np.pi / freq

    # Perform non-linearregression in order to polish the results
    y -= (max_y - amp)
    coeffs = np.empty(3)
    coeffs[0] = amp
    coeffs[1] = freq
    coeffs[2] = phase

    coeffs, _ = curve_fit(sin_regressor, X_reg, y, p0=coeffs)

    # Reenable scipy warnings
    warnings.simplefilter("default", RuntimeWarning)
    warnings.simplefilter("default", OptimizeWarning)

    return coeffs

  def __call__(self, x):
    """ Predicts the value at point x """
    return sin_regressor(x, *self.coefficients)

  def __str__(self):
    """ Creates a string representation with possible line breaks """
    code = str(self.coefficients[0]) + " * std::sin($lb$"
    code += str(self.coefficients[1]) + " * x + "
    code += str(self.coefficients[2]) + "$lb$)"

    return code
