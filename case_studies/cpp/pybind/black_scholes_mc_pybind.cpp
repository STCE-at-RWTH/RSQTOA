// Include the pybind modules
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// Include of the original model
#include "black_scholes_mc.hpp"

PYBIND11_MODULE(black_scholes_mc_binding, m) {
  m.doc() = "RSQTOA - Python binding of the original model.";
  m.def(
    "f", &f<float>,
    "RSQTOA - Generated entry point of the original model");
}
