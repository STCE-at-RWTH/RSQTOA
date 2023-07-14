from argparse import ArgumentError
import os
import yaml

# ---------------------------------------------------------------------------- #

def _get_arg_value(argv, option, default = ""):
  """Extract a command line option from argument list.

  Parameters
  ----------
  argv : list
    List of command line arguments.

  option : str
    The option which the function shall search for.

  default : str, optional
    The default value in case the option is not in the argument list.

  """
  value = ""

  tmp_option = "--" + option + "="
  found = False
  for arg in argv:
    if arg.startswith(tmp_option):
      found = True
      value = arg[len(tmp_option):]
      if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]

  if not found:
    if default != "":
      value = default
    else:
      raise RuntimeError("Expected command line option: '" + tmp_option + "'.")

  return value

def _get_arg_values(argv, option, separator=" ", default=[]):
  """Extract a command line option from argument list and separates it.

  Parameters
  ----------
  argv : list
    List of command line arguments.

  option : str
    The option which the function shall search for.

  default : str, optional
    The default value in case the option is not in the argument list.

  """
  value = _get_arg_value(argv, option, None)

  if value:
    return value.split(separator)
  else:
    return default

def _arg_exists(argv, option):
  """Check if a command line option exists in the argument list.

  Parameters
  ----------
  argv : list
    List of command line arguments.

  option : str
    The option which the function shall search for.

  """
  return ("--" + option) in argv

def _load_config_file(config_file):
  """Loads a yaml config file.

  Parameters
  ----------
  config_file : str
    The path to the config file.

  """
  yaml_config = None
  with open(config_file, 'r') as configuration:
    try:
      yaml_config = yaml.safe_load(configuration)
    except yaml.YAMLError as exc:
      raise SyntaxError(
        "Unable to parse configuration file: '" + config_file + "'.")

  return yaml_config

def _create_config(argv=None, cfg=None):
  """Creates a unified configuration from the default config, provided config
     file and options passed through command line arguments.

  Parameters
  ----------
  argv : list
    List of command line arguments.

  """
  if not (argv or cfg):
    raise ArgumentError("Neither argv nor config is given.")

  # Load default config file
  default_config_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "default.yml")
  yaml_config = _load_config_file(default_config_file)

  # Set system dependent defaults
  yaml_config["output_dir"] = os.getcwd()

  # Look for a provided config file and replace the options in the config
  if cfg:
    specialized_config_file = cfg
  else:
    specialized_config_file = _get_arg_value(argv, "config", default=None)
  if specialized_config_file:
    specialized_config = _load_config_file(specialized_config_file)
    for key in yaml_config:
      if key in specialized_config and specialized_config[key] is not None:
        yaml_config[key] = specialized_config[key]

  # Parse remaining command line arguments and replace the options in the config
  if argv:
    for key in yaml_config:
      if isinstance(yaml_config[key], list):
        yaml_config[key] = _get_arg_values(argv, key, default=yaml_config[key])
      else:
        yaml_config[key] = _get_arg_value(argv, key, default=yaml_config[key])

  return yaml_config
