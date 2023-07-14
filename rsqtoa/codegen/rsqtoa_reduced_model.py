"""rsqtoa_original_model.py

This file is part of the rsqtoa codegen package. It provides functions
which generate code involving the original model.

"""

# Import global config
import rsqtoa.codegen.rsqtoa_codegen_config as config

# Import needed generator functions
from rsqtoa.codegen.rsqtoa_original_model import _generate_model_header_name
from rsqtoa.codegen.rsqtoa_format import _indent
from rsqtoa.codegen.rsqtoa_format import _javastyle_doc
from rsqtoa.codegen.rsqtoa_format import _join_with_linebreaks

# ---------------------------------------------------------------------------- #

def _leaf_id():
  """Generate the id of the current leaf."""
  return str(config.current_subspace.leaf_id + 1)

def _leaf_amount():
  """Generate the amount of leafs."""
  return str(config.leafs)

def _sample_point(regression):
  """Generate the sample point of the regression function.

    Parameters
    ----------
    regression : QuasiSeparabilityRegression
      The regression function object.

  """
  return "x" + str(regression.regression_dim) + "_sp"

def _sample_point_definition(regression):
  """Generate the sample point definition of the regression function.

    Parameters
    ----------
    regression : QuasiSeparabilityRegression
      The regression function object.

  """
  code = _indent() + "const T " + _sample_point(regression)
  code += " = T(" + str(regression.sample_value) + ");"

  return code

def _regression_function_name(regression, ss_id):
  """Generate the name of the regression function.

    Parameters
    ----------
    regression : QuasiSeparabilityRegression
      The regression function object.
      
    ss_id : str
      The id of the subspace where the regression is valid.

  """
  if ss_id:
    return "rsqtoa_s" + "_".join(ss_id) + "_g" + str(regression.regression_dim)
  else:
    return "rsqtoa_g" + str(regression.regression_dim)

def _subspace_regression(regression):
  """Generate a single subspace regression function template.

    Parameters
    ----------
    regression : QuasiSeparabilityRegression
      The regression function object.

  """
  ss_id = config.current_subspace.get_ss_id()
  
  doc = "Regression in dimension " + str(regression.regression_dim)
  doc += " (domain: ["
  doc += str(config.current_subspace.lower) + " x "
  doc += str(config.current_subspace.upper) + "])"
  code = _javastyle_doc(doc)

  code += _indent() + "template<typename T>\n"
  code += _indent() + "static inline T "
  code += _regression_function_name(regression, ss_id)
  code += "(const T &x) {\n\n"
  code += _indent(1) + "return (\n" 
  
  config.current_indent += 2
  code += _join_with_linebreaks(str(regression).split("$lb$"), " ") + "\n"
  config.current_indent -= 2

  code += _indent(1) + ");\n\n" + _indent() + "}"

  return code

def _reduced_dimensions_name():
  """Generate the dimension name for a reduced model."""
  return config.model_function + "_reduced_dims_" + _leaf_id()

def _full_dimensions_name():
  """Generate the dimension name the full model."""
  return config.model_function + "_full_dims"

def _reduced_model_name():
  """Generate the function name for a reduced model."""
  return config.model_function + "_reduced_" + _leaf_id()

def _reduced_type_name():
  """Generate the input type name for a reduced model."""
  return config.model_function + "_reduced_input_" + _leaf_id() + "_t"

def _reduced_ranges_name():
  """Generate the reduced range type name for a reduced model."""
  return config.model_function + "_reduced_ranges_" + _leaf_id()

def _full_ranges_name():
  """Generate the full range type name for a reduced model."""
  return config.model_function + "_full_ranges_" + _leaf_id()

def _reassambled_model_name():
  """Generate the function name for the reassambled model function."""
  return config.model_function + "_reassambled_model_" + _leaf_id()

def _reduced_model_header():
  """Generate the header name for a reduced model."""
  return _generate_model_header_name() + "_rsqtoa_" + _leaf_id() + ".hpp"

def _learned_model_type():
  """Generate the template type of a learned model."""
  return "LEARNED_MODEL_" + _leaf_id()

def _learned_model_arg():
  """Generate a learned model argument in the reassemble function."""
  return "lm" + _leaf_id()

def _reduced_range():
  """Generate the reduced range for a reduced model"""
  code = _indent() + "template<typename T>\n"
  code += _indent() + "static constexpr boundaries_t<T, "
  code += _reduced_dimensions_name() + ">\n"
  code += _reduced_ranges_name() + " = {{\n"
  
  ranges = []
  for i in config.current_subspace.non_separable_dims:
    range_pair = "{" + str(config.current_subspace.lower[i])
    range_pair += ", " + str(config.current_subspace.upper[i]) + "}"
    ranges.append(range_pair)

  config.current_indent += 1
  code += _join_with_linebreaks(ranges, ", ") + "\n"
  config.current_indent -= 1
  
  code += _indent() + "}};"
  
  return code

def _full_range():
  """Generate the full range for a reduced model"""
  code = _indent() + "template<typename T>\n"
  code += _indent() + "static constexpr boundaries_t<T, "
  code += _full_dimensions_name() + ">\n"
  code += _full_ranges_name() + " = {{\n"
  
  ranges = []
  for i in range(config.full_dims):
    range_pair = "{" + str(config.current_subspace.lower[i])
    range_pair += ", " + str(config.current_subspace.upper[i]) + "}"
    ranges.append(range_pair)

  config.current_indent += 1
  code += _join_with_linebreaks(ranges, ", ") + "\n"
  config.current_indent -= 1
  
  code += _indent() + "}};"
  
  return code

def _model_reassembling():
  """Generate the optional model evaluation, if the input is in the subspace"""
  code = "if (check_bounds<T, " + _full_dimensions_name()  + ">("
  code += _full_ranges_name() + "<T>, x)) {\n"
  code += _indent(1) + "return " + _reassambled_model_name()
  code += "(x, " + _learned_model_arg() + ");\n"
  code += _indent() + "}"

  return code

# ---------------------------------------------------------------------------- #

def _generate_leaf_id():
  """Generate the id of the current leaf."""
  return _indent() + _leaf_id()

def _generate_leaf_amount():
  """Generate the id of the current leaf."""
  return _indent() + _leaf_amount()

def _generate_reduced_dimensions():
  """Generates the amount of dimensions in the reduced model."""
  code = ""
  for leaf_id in range(config.leafs):
    config.current_subspace = config.ss_tree.get_leaf(leaf_id)
    code += _indent() + "static constexpr rsqtoa_uint_t "
    code += _reduced_dimensions_name() + " = "
    code += str(len(config.ss_tree.get_leaf(leaf_id).non_separable_dims))
    code += ";\n"

  return code[:-1]

def _generate_full_dimonsions():
  """Generates the amount of dimensions in the full model."""
  return str(config.full_dims)

def _generate_reduced_x():
  """Generates the input variable in the reduced model."""
  code = ""
  if len(config.current_subspace.non_separable_dims) > 0:
    code += "x_red"

  return code

def _generate_regression_functions():
  """Generates all subspace regression function templates."""
  code = ""

  for ss in config.ss_tree.get_all_subspaces():
    config.current_subspace = ss
    for reg in ss.regressions:
      code += _subspace_regression(reg) + "\n\n"

  return code[:-2]

def _generate_sample_points():
  """Generates all regression sample points for the reduced model."""
  code = ""

  for reg in config.current_subspace.get_regressions():
    code += _sample_point_definition(reg) + "\n"

  return code[:-1]

def _generate_full_inputs():
  """Generates the inputs for the full model."""
  elements = []

  i_red = 0
  for i in range(config.full_dims):
    elem = "x_red[" + str(i_red) + "]"
    i_red += 1
    for reg in config.current_subspace.get_regressions():
      if reg.regression_dim == i:
        elem = _sample_point(reg)
        i_red -= 1
        break

    elements.append(elem)

  return _join_with_linebreaks(elements, ", ")

def _generate_reduced_inputs():
  """Generates the input vector for the reduced model."""
  elements = []

  if len(config.current_subspace.non_separable_dims) > 0:
    for i in range(config.full_dims):
      has_regression =  False
      for reg in config.current_subspace.get_regressions():
        if reg.regression_dim == i:
          has_regression =  True
          break
      if not has_regression:
        elements.append("x[" + str(i) + "]")
  else:
    elements.append("T()")

  return _join_with_linebreaks(elements, ", ")

def _generate_model_reduction():
  """Generates the model reductions."""
  code = ""

  regs, ss_ids = config.current_subspace.get_regressions_with_ss_id()
  for reg, ss_id in zip(regs, ss_ids):
    code += _indent() + "ret_value -= "
    code += _regression_function_name(reg, ss_id)
    code += "(" + _sample_point(reg) + ");\n"

  return code[:-1]

def _generate_model_reassambling():
  """Generates the reassambling of the model."""
  code = ""

  regs, ss_ids = config.current_subspace.get_regressions_with_ss_id()
  for reg, ss_id in zip(regs, ss_ids):
    code += _indent() + "ret_value += "
    code += _regression_function_name(reg, ss_id)
    code += "(x[" + str(reg.regression_dim) + "]);\n"

  return code[:-1]

def _generate_reduced_model_ranges():
  """Generate the ranges of the reduced model function."""
  code = ""
  for leaf_id in range(config.leafs):
    config.current_subspace = config.ss_tree.get_leaf(leaf_id)  
    code += _full_range() + "\n"
    code += _reduced_range() + "\n"
    
  return code[:-1]

def _generate_reduced_types():
  """Generate the input types for all reduced models."""
  code = ""
  for leaf_id in range(config.leafs):
    config.current_subspace = config.ss_tree.get_leaf(leaf_id)
    code += _indent() + "template<typename T>\n"
    code += _indent() + "using " + _reduced_type_name()
    code += " = vector_t<T, " + _reduced_dimensions_name() + ">;\n"
    
  return code[:-1]

def _generate_reduced_model_includes():
  """Generate the includes for all reduced models."""
  code = ""
  for leaf_id in range(config.leafs):
    config.current_subspace = config.ss_tree.get_leaf(leaf_id)
    code += '#include "' + _indent() + _reduced_model_header() + '"\n'

  return code[:-1]

def _generate_learned_model_types():
  """Generate the template types of all learned model."""
  code = ""
  for leaf_id in range(config.leafs):
    config.current_subspace = config.ss_tree.get_leaf(leaf_id)
    code += "typename " + _learned_model_type() + ", "

  return code[:-2]

def _generate_learned_model_args():
  """Generate all learned model arguments in the reassemble function."""
  code = ""
  for leaf_id in range(config.leafs):
    config.current_subspace = config.ss_tree.get_leaf(leaf_id)
    code += "const " + _learned_model_type() + " &"
    code += _learned_model_arg() + ", "

  return code[:-2]

def _generate_model_reassembling():
  """Generate the logic that stitches the different models together."""
  code = _indent()
  for leaf_id in range(config.leafs):
    config.current_subspace = config.ss_tree.get_leaf(leaf_id)
    code += _model_reassembling() + " else "

  return code[:-6]
