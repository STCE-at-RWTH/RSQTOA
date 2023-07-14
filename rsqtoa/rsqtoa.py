import sys
import importlib
import numpy as np

# Import global config
from rsqtoa import rsqtoa_config as config
import rsqtoa.codegen.rsqtoa_codegen_config as codegen_config
from rsqtoa import rsqtoa_statistics as stats

# Import codegen and subspace regression modules
from rsqtoa.rsqtoa_args import _create_config
from rsqtoa.rsqtoa_subspace_tree import SubspaceTree
from rsqtoa.codegen import _generate_reduced_model
from rsqtoa.model_wrapper import Model

# ---------------------------------------------------------------------------- #

def create_config(cfg=None, argv=None):
  """ Functional interface to load a new config """

  # Subspace regression
  return _create_config(cfg=cfg, argv=argv)

def create_subspace_tree(yaml_config: object, model: Model = None) -> SubspaceTree:
  """ Functional interface to create a subspace tree for a given model """

  if yaml_config["model_dir"]:
    sys.path.append(yaml_config["model_dir"])

  dims = 0
  lower = []
  upper = []

  # Import model
  if not model:
    if yaml_config["model_module"]:
      model_module = importlib.import_module(yaml_config["model_module"])
      model = getattr(model_module, yaml_config["model_function"])
    else:
      model = globals()[yaml_config["model_function"]]

    # Define domain and function
    dims = int(yaml_config["dimensions"])
    lower = np.array(yaml_config["lower_bounds"])
    upper = np.array(yaml_config["upper_bounds"])
  else:
    dims = model.dims
    lower = np.array(model.lower)
    upper = np.array(model.upper)


  # Subspace regression config
  config.epsilon = float(
    yaml_config["epsilon"])
  config.sampling_strategy = yaml_config["sampling_strategy"]
  config.regression_samples = int(
    yaml_config["regression_samples"])
  config.test_samples = int(
    yaml_config["test_samples"])
  config.test_values = int(
    yaml_config["test_values"])
  config.max_taylor_order = int(
    yaml_config["max_taylor_order"])
  config.taylor_regression_samples = int(
    yaml_config["taylor_regression_samples"])
  config.periodic_regression_samples = int(
    yaml_config["periodic_regression_samples"])
  config.exponential_regression_samples = int(
    yaml_config["exponential_regression_samples"])
  config.debug = bool(
    yaml_config["debug"])

  config.subspace_ordering=str(
    yaml_config["subspace_ordering"])
  config.subspace_depth = int(
    yaml_config["subspace_depth"])
  config.subspace_threshold = float(
    yaml_config["subspace_threshold"])
  config.subspace_needs_regression = bool(
    yaml_config["subspace_needs_regression"])
  config.subspace_selection = yaml_config["subspace_selection"]
  config.regression_selection = yaml_config["regression_selection"]
  config.subspace_expansion_steps = int(
    yaml_config["subspace_expansion_steps"])
  config.subspace_reduction_steps = int(
    yaml_config["subspace_reduction_steps"])

  # Subspace regression
  return SubspaceTree(model, dims, lower, upper)

def generate_reduced_model(ss_tree, yaml_config):
  """ Functional interface to generate C++ code with the reduced models """

  # Setup output directories
  codegen_config.output_dir = yaml_config["output_dir"]
  codegen_config.model_file = yaml_config["model_file"]
  codegen_config.model_module = yaml_config["model_module"]
  codegen_config.model_function = yaml_config["model_function"]

  # Setup model
  codegen_config.full_dims = yaml_config["dimensions"]
  codegen_config.lower = yaml_config["lower_bounds"]
  codegen_config.upper = yaml_config["upper_bounds"]

  # Reduced model codegen
  codegen_config.leafs = stats.leafs

  _generate_reduced_model(ss_tree)

  # Write epsilon file only if there was an epsilon input file
  if yaml_config["epsilon_input_file"]:
    with open(yaml_config["epsilon_output_file"], 'w') as epsilon_file:
      epsilon_file.write(str(config.epsilon))
