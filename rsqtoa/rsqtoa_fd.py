import numpy as np

# ---------------------------------------------------------------------------- #

def _fd_derivative(model, lower, upper, reg_dim, x):
  """ Return the derivative of the model at point x. """

  # Finite difference epsilon
  h = (1 + np.abs(x[reg_dim])) * np.cbrt(np.finfo(np.float32).eps)
  h = 2.0 ** np.round(np.log(h) / np.log(2.0))
  dx = 0.0
  x_tmp = np.copy(x)

  if x[reg_dim] == lower[reg_dim]:
    # Forward scheme with second order accuracy
    dx -= 3.0 * model.get_target(x_tmp)
    x_tmp[reg_dim] += h
    dx += 4.0 * model.get_target(x_tmp)
    x_tmp[reg_dim] += h
    dx -= 1.0 * model.get_target(x_tmp)
  elif x[reg_dim] == upper[reg_dim]:
    # Backward scheme with second order accuracy
    dx += 3.0 * model.get_target(x_tmp)
    x_tmp[reg_dim] -= h
    dx -= 4.0 * model.get_target(x_tmp)
    x_tmp[reg_dim] -= h
    dx += 1.0 * model.get_target(x_tmp)
  else:
    # Central scheme with second order accuracy
    x_tmp[reg_dim] += h
    dx += model.get_target(x_tmp)
    x_tmp[reg_dim] -= 2*h
    dx -= model.get_target(x_tmp)

  dx /= 2 * h
  return dx
