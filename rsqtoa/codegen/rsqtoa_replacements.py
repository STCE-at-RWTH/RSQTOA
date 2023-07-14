"""rsqtoa_replacements.py

This file is part of the rsqtoa codegen package.

"""

# Global config
import rsqtoa.codegen.rsqtoa_codegen_config as config
from rsqtoa.codegen.rsqtoa_codegen_config import TAB

# Import needed generator functions
from rsqtoa.codegen.rsqtoa_format import _indent

# Meta information
from rsqtoa.codegen.rsqtoa_meta import _generate_timestamp
from rsqtoa.codegen.rsqtoa_meta import _generate_version

# Original model
from rsqtoa.codegen.rsqtoa_original_model import _generate_model_header_name
from rsqtoa.codegen.rsqtoa_original_model import _generate_model_header_include
from rsqtoa.codegen.rsqtoa_original_model import _generate_model_function
from rsqtoa.codegen.rsqtoa_original_model import _generate_model_function_signature
from rsqtoa.codegen.rsqtoa_original_model import _generate_model_ranges

# Reduced model
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_leaf_amount
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_leaf_id
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_full_dimonsions
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_reduced_dimensions
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_reduced_x
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_regression_functions
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_sample_points
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_full_inputs
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_reduced_inputs
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_model_reduction
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_model_reassambling
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_reduced_model_ranges
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_reduced_types
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_reduced_model_includes
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_learned_model_types
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_learned_model_args
from rsqtoa.codegen.rsqtoa_reduced_model import _generate_model_reassembling

# ---------------------------------------------------------------------------- #

RSQTOA_REPLACEMENTS = {
  # Meta information
  "$VERSION$": (lambda : _generate_version()),
  "$TIMESTAMP$": (lambda : _generate_timestamp()),

  # Original model
  "$MODEL_NAME$": (
    lambda : _generate_model_header_name()),
  "$MODEL_NAME_UPPER$": (
    lambda : _generate_model_header_name().upper()),
  "$MODEL_INCLUDE$": (
    lambda : _generate_model_header_include()),
  "$MODEL_FUNCTION$": (
    lambda : _generate_model_function()),
  "$MODEL_FUNCTION_SIGNATURE$": (
    lambda : _generate_model_function_signature()),
  "$MODEL_RANGES$": (
    lambda : _generate_model_ranges()),

  # Reduced model
  "$LEAF_AMOUNT$": (
    lambda : _generate_leaf_amount()),
  "$LEAF_ID$": (
    lambda : _generate_leaf_id()),
  "$MODEL_RANGES_REDUCED$": (
    lambda : _generate_reduced_model_ranges()),
  "$REGRESSION_FUNCTIONS_CXX$": (
    lambda : _generate_regression_functions()),
  "$REDUCED_DIMS$": (
    lambda : _generate_reduced_dimensions()),
  "$REDUCED_X$": (
    lambda : _generate_reduced_x()),
  "$FULL_DIMS$": (
    lambda : _generate_full_dimonsions()),
  "$SAMPLE_POINTS$": (
    lambda : _generate_sample_points()),
  "$FULL_INPUTS$": (
    lambda : _generate_full_inputs()),
  "$REDUCED_INPUTS$": (
    lambda : _generate_reduced_inputs()),
  "$MODEL_REDUCTION$": (
    lambda : _generate_model_reduction()),
  "$MODEL_REASSAMBLING_CXX$": (
    lambda : _generate_model_reassambling()),
  "$REDUCED_TYPES$": (
    lambda : _generate_reduced_types()),
  "$REDUCED_MODEL_INCLUDES$": (
    lambda : _generate_reduced_model_includes()),
  "$LEARNED_MODEL_TYPES$": (
    lambda : _generate_learned_model_types()),
  "$LEARNED_MODELS$": (
    lambda : _generate_learned_model_args()),
  "$LEARNED_MODEL_SELECTION$": (
    lambda : _generate_model_reassembling())

}

# ---------------------------------------------------------------------------- #

def _apply_replacements(template):
  """Applies replacements.

  Parameters
  ----------
  template : str
    Loaded code template.

  """
  code = template

  for key in RSQTOA_REPLACEMENTS:
    idx = code.find(key)
    while idx >= 0:
      config.current_indent = 0
      while code[idx-len(TAB):idx] == TAB:
        config.current_indent += 1
        idx -= len(TAB)

      code = code.replace(_indent() + key, RSQTOA_REPLACEMENTS[key]())
      idx = code.find(key)

  return code
