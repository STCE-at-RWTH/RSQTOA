"""rsqtoa_codegen_config.py

This file is part of the rsqtoa codegen package. It provides global a place
for global variables and constants.

"""

import os

# ---------------------------------------------------------------------------- #

MAJOR_VERSION = "0"
"""Major version number."""

MINOR_VERSION = "1"
"""Minor version number."""

REVISION = "0"
"""Revision number."""

TAB = "  "
"""Tab string."""

LINE_WIDTH = 80
"""Maximum amount of characters per line."""

TEMPLATE_DIR = os.path.join(
  os.path.dirname(os.path.realpath(__file__)), "templates")
"""Path to the template directory."""

# ---------------------------------------------------------------------------- #

tmp_dir = ""
"""Path to the temporary directory directory."""

output_dir = ""
"""Path to the output directory."""

current_indent = 0
"""Current indentation."""

model_file = ""
"""Name of the header file of the original model."""

model_module = ""
"""Name of the python module of the original model."""

model_function = ""
"""Name of the original model."""

template_type = ""
"""Typename of the template type."""

# ---------------------------------------------------------------------------- #

full_dims = 0
""" Full amount of dimensions """

ss_tree = None
""" Subspace regression tree """

current_leaf = None
""" Current leaf subspace """

leafs = 0
""" Amount of subspaces """

# ---------------------------------------------------------------------------- #
