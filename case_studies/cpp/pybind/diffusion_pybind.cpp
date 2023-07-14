// Include the pybind modules
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// Include of the original model
#include "diffusion.hpp"

PYBIND11_MODULE(diffusion_binding, m) {
  m.doc() = "RSQTOA - Python binding of the original model.";
  m.def(
    "diffusion", &diffusion<float>,
    "RSQTOA - Generated entry point of the original model");
}
