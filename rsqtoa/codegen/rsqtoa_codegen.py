"""rsqtoa_replacements.py

This file is part of the rsqtoa codegen package. It provides functions
which trigger the code generation.

"""

# System imports
import os

# Import global config
import rsqtoa.codegen.rsqtoa_codegen_config as config

# Import needed generator functions
from rsqtoa.codegen.rsqtoa_original_model import _generate_model_header_name
from rsqtoa.codegen.rsqtoa_reduced_model import _leaf_id
from rsqtoa.codegen.rsqtoa_replacements import _apply_replacements

# ---------------------------------------------------------------------------- #

def _generate_file(template_file, output_file_suffix=None):
  """Generates a file by applying the replacements to a template file.

  Parameters
  ----------
  template_file : str
    Path to the template file.

  output_file : str
    Path to the output file.
      
  """
  # Template file for the cpp binding
  template_file = os.path.join(
    config.TEMPLATE_DIR, template_file)

  # Output file for the cpp binding
  output_name = _generate_model_header_name() + "_rsqtoa"
  if output_file_suffix:
    output_name += "_" + output_file_suffix
  output_name += ".hpp"
  output_file = os.path.join(config.output_dir, output_name)

  # Create directory for generated files
  if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)
  
  file_content = ""

  # Generate file content
  with open(template_file, "r") as tpf:
    template = tpf.read()
    file_content = _apply_replacements(template)

  # write file content
  with open(output_file, "w+") as of:
    of.write(file_content)

def _generate_model_binding(model_module, model_function, template_type):
  """Generates the binding of a c++ model.

  Parameters
  ----------
  model_module : str
    Name of the model header.

  model_function : str
    Name of the model function.

  template_type : str
    Typename of the template type.
      
  """
  # Init config
  config.model_module = model_module
  config.model_function = model_function
  config.template_type = template_type

  # Template file for the cpp binding
  template_file = os.path.join(config.TEMPLATE_DIR, "cpp_binding.tp")

  # Output file for the cpp binding
  output_name = _generate_model_header_name() + "_pybind.cpp"
  output_file = os.path.join(config.tmp_dir, output_name)

  # Create directory for generated files
  if not os.path.exists(config.tmp_dir):
    os.makedirs(config.tmp_dir)
  
  # Generate file
  _generate_file(template_file, output_file)

  return output_file

def _generate_reduced_model(ss_tree):
  """Generates the reduced model.

  Parameters
  ----------
  ss_tree : SubspaceTree
    The subspace regression tree.
      
  """
  # Init config
  config.ss_tree = ss_tree
  
  # Generate file
  _generate_file("reduced_model_cxx_header.tp")
  _generate_file("reduced_model_cxx_regressions.tp", "regressions")
  _generate_file("model_info_cxx.tp", "types")
  for i in range(config.leafs):
    config.current_subspace = ss_tree.get_leaf(i)
    _generate_file("reduced_model_cxx.tp", _leaf_id())
