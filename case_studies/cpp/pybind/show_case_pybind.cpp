// Include the pybind modules
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// Include of the original model
#include "show_case.hpp"

PYBIND11_MODULE(show_case_binding, m) {
  m.doc() = "RSQTOA - Python binding of the original model.";
  m.def(
    "f", &f<float>,
    "RSQTOA - Generated entry point of the original model");
}
