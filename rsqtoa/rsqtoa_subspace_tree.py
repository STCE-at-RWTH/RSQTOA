import copy
import numpy as np
import itertools
from datetime import datetime

# Import regression classes
from rsqtoa.rsqtoa_taylor_regression import QuasiTaylorRegression
from rsqtoa.rsqtoa_periodic_regression import QuasiPeriodicRegression
from rsqtoa.rsqtoa_exponential_regression import QuasiExponentialRegression
from rsqtoa.rsqtoa_sampling import RandomSampler, LatinHypercubeSampler, GridSampler

from rsqtoa.fapprox.rsqtoa_fapprox import denormalize_data

# Import global config and statistics
from rsqtoa import rsqtoa_config as config
from rsqtoa import rsqtoa_statistics as stats

# All regression types we want to use in our analysis
REGRESSION_TYPES = [
  QuasiTaylorRegression, QuasiPeriodicRegression,
  QuasiExponentialRegression]

# ---------------------------------------------------------------------------- #

class SubspaceNode:
  def __init__(self, model, dims, lower, upper, parent=None, id=0):
    """ Constructor of the subspace node """
    self.parent = parent
    self.children = []
    self.split_dim = -1
    self.split_value = np.NaN
    self.is_leaf = False

    # Models
    self.original_model = model
    self.approximated_model = None
    self.denormalization_idx = -1

    # Bounds of the subspace
    self.dims = dims
    self.lower = lower.astype(float)
    self.upper = upper.astype(float)

    if config.sampling_strategy == 'random':
      self.sampler = RandomSampler(model, dims, lower, upper)
    elif config.sampling_strategy == 'dataset':
      self.sampler = GridSampler(model, dims, lower, upper)
    else:
      self.sampler = LatinHypercubeSampler(model, dims, lower, upper)

    # Set subspace id
    self.id = id
    self.leaf_id = -1

    # Regression configurations
    self.regressions = []
    self.offset = None
    if parent:
      self.regression_dims = copy.deepcopy(parent.non_separable_dims)
      self.eps = parent.remaining_eps
      self.remaining_eps = parent.remaining_eps
    else:
      self.regression_dims = list(range(dims))
      self.eps = config.epsilon
      self.remaining_eps = config.epsilon

    self.non_separable_dims = copy.deepcopy(self.regression_dims)
    self.regression_errors = {
      d: {r: np.inf for r in REGRESSION_TYPES} for d in self.regression_dims}

    if config.debug:
      print(
        "\nNew Subspace Node: lower={0} upper={1}".format(
          self.lower, self.upper
        ), flush=True)

  def generate_regressions(self, regression_types=REGRESSION_TYPES):
    """ Tries to generate subspace regressions for the remaining dimensions """

    # Generate regressions
    tmp_regressions = [[] for _ in range(self.dims)]
    for reg_dim in self.regression_dims:
      # Iterate over regression models
      for Regression in regression_types:
        reg = Regression(
          self.original_model, self.dims, self.lower, self.upper, reg_dim)
        if reg.try_fit(self.eps):
          tmp_regressions[reg_dim].append(reg)
          if config.regression_selection == "greedy":
            break

        self.regression_errors[reg_dim][Regression] = reg.max_error

    # Remove non succesful dimensions
    successful_dims = self.dims - tmp_regressions.count([])
    tmp_regressions = [reg for dim in tmp_regressions for reg in dim]

    # Chose best combination of regressions
    for k in reversed(range(successful_dims)):
      regression_combinations = itertools.combinations(tmp_regressions, k+1)
      min_eps = np.Inf
      best_combination = None
      for comb in regression_combinations:
        # Filter out permutations with duplicate dimensions
        if len(set(map(lambda reg: reg.regression_dim, comb))) != len(comb):
          continue

        tmp_eps = np.sum(list(map(lambda reg: reg.max_error, comb)))
        if tmp_eps < min_eps:
          min_eps = tmp_eps
          best_combination = comb

      if min_eps <= self.eps:
        self.regressions = list(best_combination)
        self.remaining_eps = self.eps - min_eps
        break

    for reg in self.regressions:
      self.regression_errors.pop(reg.regression_dim)
      self.non_separable_dims.remove(reg.regression_dim)

    self.is_leaf = len(self.non_separable_dims) == 0

    # Calculate offset for subspaces with full regression
    if self.is_leaf:
      X, y = self.sampler.sample_domain(config.regression_samples)
      Y = self.evaluate_reduced_model(X.transpose())
      self.offset = np.sum(Y) / config.regression_samples
      if config.debug:
        print("Calcualting offset for subspace:  " + str(self.offset))

  def generate_subspaces(self):
    """ Tries to find subspaces with successful regressions """
    found_new_subspaces = False

    # If this node is already divided, pass the call onto its children
    if self.children or self.is_leaf:
      for child in self.children:
        found_new_subspaces |= child.generate_subspaces()
      return found_new_subspaces

    subspaces1 = []
    subspaces2 = []
    new_subspace1 = None
    new_subspace2 = None

    # Sort the dimensions for the subspace division (only with greedy selection)
    split_dims = range(self.dims)
    if config.subspace_selection == "greedy":
      if config.subspace_ordering == "min_error":
        split_dims = sorted(
          split_dims, key=lambda d: np.min(self.regression_errors[d].values()))
      elif config.subspace_ordering == "max_error":
        split_dims = sorted(
          split_dims, key=lambda d: np.min(self.regression_errors[d].values()),
          reverse=True)
      elif config.subspace_ordering == "area":
        split_dims = sorted(
          split_dims, key=lambda d: self.upper[d] - self.lower[d],
          reverse=True)

    # Perform subspace division
    for dim in range(self.dims):
      node1, node2, eff1, eff2 = self.split_dimension(dim)

      if config.subspace_selection == "greedy" and (eff1 > 1 or eff2 > 1):
        new_subspace1 = node1
        new_subspace2 = node2
        break
      else:
        subspaces1.append(node1)
        subspaces2.append(node2)

    # Choose best subspaces
    if config.subspace_selection == "best":
      max_eff = 1
      for ss1, ss2 in zip(subspaces1, subspaces2):
        eff = ss1.calculate_effectiveness() * ss2.calculate_effectiveness()
        if eff > max_eff:
          new_subspace1 = ss1
          new_subspace2 = ss2
          eff1 = ss1.calculate_effectiveness()
          eff2 = ss2.calculate_effectiveness()
          max_eff = eff

    # Choose best subspaces with no regressions
    if not (new_subspace1 or new_subspace2):
      if not config.subspace_needs_regression:
        min_error = np.Inf
        for ss1, ss2 in zip(subspaces1, subspaces2):
          err1 = ss1.get_smallest_regression_error()
          err2 = ss2.get_smallest_regression_error()
          err = err1 * err2
          if err < min_error:
            new_subspace1 = ss1
            new_subspace2 = ss2
            min_error = err

    # Add new subspaces
    if new_subspace1 and new_subspace2:
      if config.debug:
        print("Adding new subspaces ...")
        print("... Effectiveness: " + str(eff1) + " " + str(eff2))
      self.children.append(new_subspace1)
      self.children.append(new_subspace2)
      found_new_subspaces = True
    else:
      if config.debug:
        print("Couldn't find new subspaces.")

    self.is_leaf = not found_new_subspaces

    return found_new_subspaces

  def get_smallest_regression_error(self):
    """ Return the smallest regression error. """
    min_error = np.Inf
    for dim in self.regression_errors.values():
      for error in dim.values():
        if error < min_error:
          min_error = error

    return min_error

  def split_dimension(self, dim):
    """ Search for new subspaces """
    # Subspace boundaries
    self.split_dim = dim
    lb1 = np.copy(self.lower)
    lb2 = np.copy(self.lower)
    ub1 = np.copy(self.upper)
    ub2 = np.copy(self.upper)

    # Generate the subspace node from the current config
    def generate_nodes():
      node1 = SubspaceNode(self.original_model, self.dims, lb1, ub1, self, 1)
      node1.generate_regressions()
      eff1 = node1.calculate_effectiveness()
      node2 = SubspaceNode(self.original_model, self.dims, lb2, ub2, self, 2)
      node2.generate_regressions()
      eff2 = node2.calculate_effectiveness()
      return node1, node2, eff1, eff2

    # Initial split
    self.split_value = (self.upper[dim] + self.lower[dim]) / 2.0
    lb2[dim] = self.split_value
    ub1[dim] = self.split_value

    if config.debug:
      print("\nSplitting dim " + str(dim) + " at " + str(self.split_value))

    node1, node2, eff1, eff2 = generate_nodes()
    print("Effectiveness: " + str(eff1) + " " + str(eff2))

    # Search for at least one effective subspace (reduce first subspace)
    for i in range(config.subspace_reduction_steps):
      if eff1 > 1 or eff2 > 1:
        break

      # Shift split value to 'the left'
      self.split_value = (self.split_value + self.lower[dim]) / 2.0
      if self.split_value - self.lower[dim] < config.subspace_threshold:
        break

      lb2[dim] = self.split_value
      ub1[dim] = self.split_value

      if config.debug:
        print("\nLeft reduction step " + str(i+1) + ":")
        print("\nSplitting dim " + str(dim) + " at " + str(self.split_value))

      node1, node2, eff1, eff2 = generate_nodes()
      print("Effectiveness: " + str(eff1) + " " + str(eff2))

    # Search for at least one effective subspace (reduce second subspace)
    for i in range(config.subspace_reduction_steps):
      if eff1 > 1 or eff2 > 1:
        break

      # Shift split value to 'the right'
      self.split_value = (self.upper[dim] + self.split_value) / 2.0
      if self.upper[dim] - self.split_value < config.subspace_threshold:
        break

      lb2[dim] = self.split_value
      ub1[dim] = self.split_value

      if config.debug:
        print("\nRight reduction step " + str(i+1) + ":")
        print("\nSplitting dim " + str(dim) + " at " + str(self.split_value))

      node1, node2, eff1, eff2 = generate_nodes()
      print("Effectiveness: " + str(eff1) + " " + str(eff2))

    # Expand left subspace if it is effective
    if eff1 > 1:
      left_search_bound = self.split_value
      right_search_bound = self.upper[dim]

      for i in range(config.subspace_expansion_steps):
        # Cancel expansion if both subspaces are effective
        # if eff1 > 1 and eff2 > 1:
        #   break

        # Shift split value
        self.split_value = (left_search_bound + right_search_bound) / 2.0
        lb2[dim] = self.split_value
        ub1[dim] = self.split_value
        if config.debug:
          print("\nLeft expansion step " + str(i+1) + ":")
          print("\nSplitting dim " + str(dim) + " at " + str(self.split_value))

        tmp_node1, tmp_node2, tmp_eff1, tmp_eff2 = generate_nodes()

        # If the efficiency didn't regress, save new subspaces
        if tmp_eff1 >= eff1 or (tmp_eff1 > 1 and tmp_eff2 > 1):
          left_search_bound = self.split_value
          node1, node2, eff1, eff2 = tmp_node1, tmp_node2, tmp_eff1, tmp_eff2
          if config.debug:
            print("Success - new effectiveness: " + str(eff1))
        else:
          right_search_bound = self.split_value
          if config.debug:
            print("Failure - keep old effectiveness: " + str(eff1))

    # Expand right subspace if it is effective
    if eff2 > 2:
      left_search_bound = self.lower[dim]
      right_search_bound = self.split_value

      for i in range(config.subspace_expansion_steps):
        # Cancel expansion if both subspaces are effective
        # if eff1 > 1 and eff2 > 1:
        #   break

        # Shift split value
        self.split_value = (left_search_bound + right_search_bound) / 2.0
        lb2[dim] = self.split_value
        ub1[dim] = self.split_value
        if config.debug:
          print("\nRight expansion step " + str(i+1) + ":")
          print("\nSplitting dim " + str(dim) + " at " + str(self.split_value))

        tmp_node1, tmp_node2, tmp_eff1, tmp_eff2 = generate_nodes()

        # If the efficiency didn't regress, save new subspaces
        if tmp_eff2 >= eff2 or (tmp_eff1 > 1 and tmp_eff2 > 1):
          right_search_bound = self.split_value
          node1, node2, eff1, eff2 = tmp_node1, tmp_node2, tmp_eff1, tmp_eff2
          if config.debug:
            print("Success - new effectiveness: " + str(eff2))
        else:
          left_search_bound = self.split_value
          if config.debug:
            print("Failure - keep old effectiveness: " + str(eff2))

    return node1, node2, eff1, eff2

  def calculate_effectiveness(self):
    """ Calculate effectiveness of the model reduction """

    # Relative amount of separable dimensions
    reduced_dims_factor = len(self.regression_dims)
    reduced_dims_factor -= len(self.non_separable_dims)
    reduced_dims_factor /= len(self.regression_dims)

    # Quick return if non of the dimension was separable
    if reduced_dims_factor == 0:
      return 1

    # Relative volume of the current subdomain compared to the whole domain
    area_factor = self.calculate_subspace_volume()
    area_factor /= self.calculate_root_volume()

    # Factor which considers the remaining error threshold compared to the
    # original threshold.
    reduced_eps_factor = (self.eps - self.remaining_eps) / self.eps
    if reduced_eps_factor == 0 or len(self.non_separable_dims) == 0:
      reduced_eps_factor = np.finfo(np.float32).eps

    # Calculate effectiveness of the model reduction in the current subdomain
    return 2 + int(area_factor * reduced_dims_factor / reduced_eps_factor)

  def get_leaf(self, leaf_id):
    """ Returns the leaf with the given leaf id """
    if self.leaf_id == leaf_id:
      return self

    for child in self.children:
      tmp = child.get_leaf(leaf_id)
      if tmp:
        return tmp

    return None

  def get_leafs(self):
    """ Returns all leaf nodes """
    if self.is_leaf:
      return [self]

    leafs = []
    for child in self.children:
      leafs += child.get_leafs()

    return leafs

  def get_ss_id(self):
    """ Returns the unique subspace id """
    if self.parent:
      return self.parent.get_ss_id() + [str(self.id)]
    return []

  def get_regressions(self):
    """ Returns a list of regressions for the current subspace """
    reg = []
    if self.parent:
      reg += self.parent.get_regressions()
    reg += self.regressions
    return reg

  def get_regressions_with_ss_id(self):
    """ Returns a list of regressions with their subspace ids """
    reg = []
    ss_id = []
    if self.parent:
      parent_reg, parent_ss_id = self.parent.get_regressions_with_ss_id()
      reg += parent_reg
      ss_id += parent_ss_id
    reg += self.regressions
    ss_id += [self.get_ss_id()] * len(self.regressions)
    return reg, ss_id

  def get_all_subspaces(self):
    """ Returns a list of all subspaces """
    ss = [self]
    for child in self.children:
      ss += child.get_all_subspaces()
    return ss

  def has_no_new_regressions(self):
    has_no_regressions = len(self.regressions) == 0
    for child in self.children:
      has_no_regressions &= child.has_no_new_regressions()
    return has_no_regressions

  def identify_leafs(self):
    """ Assign leaf ids and remove unnecessary subspaces """
    children_are_unnecessary = True
    for child in self.children:
      children_are_unnecessary &= child.has_no_new_regressions()

    # Remove childeren if they don't add any new regressions
    if children_are_unnecessary:
      self.children.clear()
      self.is_leaf = True

    if self.is_leaf:
      self.leaf_id = stats.leafs
      stats.leafs += 1
      stats.average_dim_reduction += self.dims - len(self.non_separable_dims)
      if self.remaining_eps < stats.remaining_epsilon:
        stats.remaining_epsilon = self.remaining_eps
    else:
      for child in self.children:
        child.identify_leafs()

  def calculate_subspace_volume(self):
    """ Calculate the volume of the current subdomain """
    volume = 1
    for dim in range(self.dims):
      volume *= (self.upper[dim] - self.lower[dim])
    return volume

  def calculate_root_volume(self):
    """ Calculate the volume of the entire domain """
    if self.parent:
      return self.parent.calculate_root_volume()
    return self.calculate_subspace_volume()

  def _check_bounds(self, x):
    """ Check if the point is inside the bounds. """
    output_dims = list(np.shape(x))
    output_dims = output_dims[1:]
    inside = np.full(tuple(output_dims), True, dtype=bool)
    for dim in range(self.dims):
      inside = np.logical_and(inside, np.logical_and(
        x[dim, :] >= self.lower[dim], x[dim, :] <= self.upper[dim]))

    return inside

  def evaluate_reduced_model(self, x):
    """ Evaluates the reduced model """
    output_dims = list(np.shape(x))
    output_dims = output_dims[1:]
    y = np.zeros(shape=tuple(output_dims))

    inside_points = self._check_bounds(x)
    if self.is_leaf:
      if np.ndim(x) == 1:
        x_tmp = np.copy(x).reshape((1, len(x)))
      else:
        x_tmp = np.copy(x)

      for reg in self.get_regressions():
        x_tmp[reg.regression_dim, :][inside_points] = reg.sample_value
        y[inside_points] -= reg(reg.sample_value)

      try:
        y[inside_points] += self.original_model.get_target(x_tmp[:, inside_points])
      except:
        it = np.nditer(inside_points, flags=['multi_index'])
        for x in it:
          if inside_points[it.multi_index]:
            y[it.multi_index] += self.original_model.get_target(x_tmp[:, it.multi_index])

      if self.offset is not None:
        y[inside_points] -= self.offset
    else:
      overlap = np.zeros_like(self._check_bounds(x), dtype=int)
      for child in self.children:
        child_check = child._check_bounds(x)
        if child_check.any():
          overlap += child_check
          y += child.evaluate_reduced_model(x)

      y[overlap > 0] /= overlap[overlap > 0]

    return y

  def evaluate_reassembled_model(self, x):
    """ Evaluates the reassembled model """
    output_dims = list(np.shape(x))
    output_dims = output_dims[1:]
    y = np.zeros(shape=tuple(output_dims))

    inside_points = self._check_bounds(x)
    if self.is_leaf:
      if np.ndim(x) == 1:
        x_tmp = np.copy(x).reshape((1, len(x)))
      else:
        x_tmp = np.copy(x)

      for reg in self.get_regressions():
        y[inside_points] += reg(x_tmp[reg.regression_dim, :][inside_points])

      if self.offset is not None:
        y[inside_points] += self.offset
      else:
        # Evaluate the approximated model. Use the reduced model if there is
        # no approximation yet
        if self.approximated_model:
          x_tmp = np.copy(x)
          for i in reversed(range(self.dims)):
            if i not in self.non_separable_dims:
              x_tmp = np.delete(x_tmp, i, 0)

          y[inside_points] += denormalize_data(
            self.approximated_model.predict(
              x_tmp.transpose()).squeeze().reshape(np.shape(y))[inside_points],
            self.denormalization_idx)
        else:
          y[inside_points] += self.evaluate_reduced_model(x[:, inside_points])
    else:
      overlap = np.zeros_like(self._check_bounds(x), dtype=int)
      for child in self.children:
        child_check = child._check_bounds(x)
        if child_check.any():
          overlap += child_check
          y += child.evaluate_reassembled_model(x)

      y[overlap > 0] /= overlap[overlap > 0]

    return y

  def get_sample_coefficient(self):
    """ Calculate the percentual amount of samples for this domain """
    reduced_dims_factor = len(self.non_separable_dims)
    reduced_dims_factor /= self.dims

    area_factor = self.calculate_subspace_volume()
    area_factor /= self.calculate_root_volume()

    return reduced_dims_factor

  def sample_domain(self, n):
    """ Sample the current domain """
    return self.sampler.sample_domain(n)

  def print_info(self):
    print()
    print("ID={}".format(self.get_ss_id()))
    print("Leaf={}".format(self.is_leaf))
    print("Subdomain lower={}".format(self.lower))
    print("Subdomain upper={}".format(self.upper))
    print("Subdomain split_dim={}".format(self.split_dim))
    print("Subdomain split_value={}".format(self.split_value))
    print("Regressions:")
    for reg in self.regressions:
      print(str(reg).replace("x", "x{}".format(reg.regression_dim)))
      print("Sample value={}".format(reg.sample_value))
    print("Subdomain offset={}".format(self.offset))

    for child in self.children:
      child.print_info()

class SubspaceTree:
  def __init__(self, model, dims, lower, upper):
    """ Constructor of the subspace tree """
    self.reset_stats()
    stats.start_time = datetime.now()
    self.model = model
    self.root = SubspaceNode(model, dims, lower, upper)
    self.root.generate_regressions()

    for i in range(config.subspace_depth):
      print("\nSubspace division depth: " + str(i+1))
      if not self.root.generate_subspaces():
        print("Couldn't find any new subspaces.")
        break

    # Assign leaf ids and simplify subspace tree if possible
    self.root.identify_leafs()
    stats.end_time = datetime.now()
    stats.average_dim_reduction /= stats.leafs

    if config.debug:
      print("\n\nRSQTOA Statistics:")
      print("Elapsed time: " + str(stats.end_time - stats.start_time))
      print("Samples: " + str(stats.amount_of_samples))
      print("Remaining epsilon: " + str(stats.remaining_epsilon))
      print("Amount of subspaces: " + str(stats.leafs))
      print("Average reduced dimensions: " + str(stats.average_dim_reduction))

  def get_leaf(self, leaf_id):
    """ Returns the leaf with the given id """
    return self.root.get_leaf(leaf_id)

  def get_leafs(self):
    """ Returns all leafs """
    return self.root.get_leafs()

  def get_all_subspaces(self):
    """ Returns a list of all subspaces """
    return self.root.get_all_subspaces()

  def evaluate_reduced_model(self, x):
    """ Evaluates the reduced model """
    if self.root._check_bounds(x).all():
      return self.root.evaluate_reduced_model(x)

    raise RuntimeError("Input x={0} out of bounds!".format(x))

  def evaluate_reassembled_model(self, x):
    """ Evaluates the reduced model """
    if self.root._check_bounds(x).all():
      return self.root.evaluate_reassembled_model(x)

    raise RuntimeError("Input x={0} out of bounds!".format(x))

  def reset_stats(self):
    """ Reset statistics """
    stats.leafs = 0
    stats.remaining_epsilon = np.Inf
    stats.start_time = None
    stats.end_time = None
    stats.amount_of_samples = 0
    stats.average_dim_reduction = 0.0

  def print_info(self):
    self.root.print_info()
