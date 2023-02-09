#include <torch/script.h>
#include "cuda_common.h"
#include "ops.h"

#define BLOCK_SIZE 128

namespace bifeat {

template <typename IntType, int TILE_SIZE>
__global__ void _UnpackBitsKernel(
    const int64_t num_items, const int64_t packed_size,
    const int64_t packed_data_dim, const int64_t out_data_dim,
    const int64_t nbits, const int64_t *const width, const IntType *const input,
    IntType *const output) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * TILE_SIZE;
  const int64_t last_node =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_items);

  while (out_node < last_node) {
    for (int idx = threadIdx.x; idx < out_data_dim; idx += BLOCK_SIZE) {
      int idx_in_input = idx;
      int chunk_idx = 0;

      while (idx_in_input >= width[chunk_idx]) {
        idx_in_input -= width[chunk_idx];
        chunk_idx += 1;
      }

      output[out_node * out_data_dim + idx] =
          input[out_node * packed_data_dim + idx_in_input];
      output[out_node * out_data_dim + idx] >>=
          nbits * (packed_size - chunk_idx - 1);
      output[out_node * out_data_dim + idx] &= (1 << nbits) - 1;
    }

    out_node += 1;
  }
}

torch::Tensor UnpackBits(torch::Tensor input_tensor, int64_t out_dim,
                         int64_t packed_size, int64_t nbits) {
  CHECK_CUDA(input_tensor);
  BIFEAT_INT_TYPE_SWITCH(input_tensor.dtype(), IntType, {
    int64_t num_items = input_tensor.size(0);
    torch::Tensor output =
        torch::empty({num_items, out_dim}, input_tensor.options());
    torch::Tensor width_log = torch::empty(
        packed_size,
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    using it = thrust::counting_iterator<int64_t>;
    thrust::for_each(
        thrust::device, it(0), it(packed_size),
        [packed_size = packed_size, dim = out_dim,
         buff = width_log.data_ptr<int64_t>()] __device__(int64_t i) mutable {
          buff[i] = int64_t((dim - i - 1) / packed_size) + 1;
        });

    constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
    _UnpackBitsKernel<IntType, TILE_SIZE><<<grid, block>>>(
        num_items, packed_size, input_tensor.size(1), out_dim, nbits,
        width_log.data_ptr<int64_t>(), input_tensor.data_ptr<IntType>(),
        output.data_ptr<IntType>());

    return output;
  });

  return torch::Tensor();
}

}  // namespace bifeat