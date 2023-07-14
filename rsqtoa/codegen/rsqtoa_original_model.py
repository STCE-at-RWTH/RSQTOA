"""rsqtoa_original_model.py

This file is part of the rsqtoa codegen package. It provides functions
which generate code involving the original model.

"""
from pathlib import Path

# Import global config
import rsqtoa.codegen.rsqtoa_codegen_config as config

# Import needed generator functions
from rsqtoa.codegen.rsqtoa_format import _indent
from rsqtoa.codegen.rsqtoa_format import _join_with_linebreaks

# ---------------------------------------------------------------------------- #

def _generate_model_header_name():
  """Generate the include of the model header."""
  return Path(config.model_file).stem

def _generate_model_header_include():
  """Generate the include of the model header."""
  return _indent() + _generate_model_header_name() + ".hpp"

def _generate_model_function():
  """Generate the name of the model function."""
  return _indent() + config.model_function

def _generate_model_function_signature():
  """Generate the name of the model function."""
  code = _indent() + config.model_function
  if config.template_type:
    code += "<rsqtoa::DCO_Type<" + config.template_type + ">>"

  return code

def _generate_model_ranges():
  """Generate the ranges of the model function."""
  code = ""

  ranges = []
  for i in range(config.full_dims):
    range_pair = "{" + str(config.lower[i]) + ", " + str(config.upper[i]) + "}"
    ranges.append(range_pair)
  
  code += _join_with_linebreaks(ranges, ", ")
  return code


