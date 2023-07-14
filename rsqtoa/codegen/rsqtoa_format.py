"""rsqtoa_format.py

This file is part of the rsqtoa codegen package. It provides functions
which helps with formatting the generated code.

"""

# Import global config
import rsqtoa.codegen.rsqtoa_codegen_config as config

# ---------------------------------------------------------------------------- #

def _indent(offset = 0):
  """Generate indentation.

    Parameters
    ----------
    offset : int
      Offset from the current indentation. Defaults to zero.

  """
  return config.TAB * (config.current_indent + offset)

def _javastyle_doc(doc_text):
  """Generate multiline java-style doc.

    Parameters
    ----------
    doc_text : str
      Text inside the java-style doc.

  """
  boundary = "*" * (79 - len(_indent()))
  code = _indent() + "/" + boundary + "\n"

  for line in doc_text.split("\n"):
    code += _indent() + " * " + line + "\n"

  code += _indent() + boundary + "/\n"

  return code

def _join_with_linebreaks(text_elements, separator):
  """Joins a list of text elements with a separator obeying the line width.

  Parameters
  ----------
  text_elements : list
    List of text elements.

  separator : str
    Separator between elements.

  """
  code = _indent()

  ll = 0
  for elem in text_elements:
    add_text = str(elem) + separator
    if ll + len(add_text) > config.LINE_WIDTH - len(_indent()):
      code += "\n" + _indent()
      ll = 0
    code += add_text
    ll += len(add_text)

  return code[:-len(separator)]

