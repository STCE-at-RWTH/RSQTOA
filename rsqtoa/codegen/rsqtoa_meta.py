"""rsqtoa_meta.py

This file is part of the rsqtoa codegen package. It provides functions
which generate meta information.

"""

# System imports
from datetime import datetime

# Import global config
from rsqtoa.codegen.rsqtoa_codegen_config import MAJOR_VERSION
from rsqtoa.codegen.rsqtoa_codegen_config import MINOR_VERSION
from rsqtoa.codegen.rsqtoa_codegen_config import REVISION

# Import needed generator functions
from rsqtoa.codegen.rsqtoa_format import _indent

# ---------------------------------------------------------------------------- #

def _generate_version():
  """Generate version string."""
  return _indent() + '.'.join([MAJOR_VERSION, MINOR_VERSION, REVISION])

def _generate_timestamp():
  """Generate timestamp."""
  return _indent() + str(datetime.now().replace(microsecond=0))
