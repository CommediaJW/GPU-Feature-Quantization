#ifndef BIFEAT_OPS_H_
#define BIFEAT_OPS_H_

#include <torch/script.h>

namespace bifeat {
torch::Tensor UnpackBits(torch::Tensor input_tensor, int64_t out_dim,
                         int64_t packed_size, int64_t nbits);
torch::Tensor DecompressVQ(torch::Tensor compressed_tensor,
                           torch::Tensor code_books, int64_t out_dim);

}  // namespace bifeat

#endif