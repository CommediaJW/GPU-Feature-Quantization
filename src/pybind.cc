#include <torch/custom_class.h>
#include <torch/script.h>

#include "ops.h"

using namespace bifeat;

TORCH_LIBRARY(bifeat_ops, m) {
  m.def("_CAPI_unpack_bits", &UnpackBits)
      .def("_CAPI_vq_decompress", &DecompressVQ);
}