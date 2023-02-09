#ifndef BIFEAT_OPS_H_
#define BIFEAT_OPS_H_

#include <torch/script.h>

namespace bifeat {
torch::Tensor UnpackBits(torch::Tensor input_tensor, int64_t out_dim,
                         int64_t packed_size, int64_t nbits);

}  // namespace bifeat

#endif