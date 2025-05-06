#include <torch/extension.h>
#include "compute_func.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_func", &ComputeFuncCPP);
  m.def("compute_func_backward", &ComputeFuncBackwardCPP);
}