#include <torch/script.h>
#include "cuda_common.h"
#include "ops.h"

#define BLOCK_SIZE 128

namespace bifeat {

template <typename IdType, typename ValueType, int TILE_SIZE>
__global__ void _DecompressVQKernel(
    const int64_t num_items, const int64_t num_parts, const int64_t length,
    const int64_t width, const int64_t out_dim, const IdType *const compressed,
    const ValueType *const code_books, ValueType *const output) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t curr_item = blockIdx.x * TILE_SIZE;
  const int64_t last_item =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_items);

  while (curr_item < last_item) {
    for (int idx = threadIdx.x; idx < out_dim; idx += BLOCK_SIZE) {
      int part_idx = idx / width;
      int idx_in_part = idx % width;
      IdType pos = compressed[curr_item * num_parts + part_idx];
      output[curr_item * out_dim + idx] =
          code_books[part_idx * length * width + pos * width + idx_in_part];
    }

    curr_item += 1;
  }
}

torch::Tensor DecompressVQ(torch::Tensor compressed_tensor,
                           torch::Tensor code_books, int64_t out_dim) {
  CHECK_CUDA(compressed_tensor);
  BIFEAT_INT_TYPE_SWITCH(compressed_tensor.dtype(), IdType, {
    BIFEAT_VALUE_TYPE_SWITCH(code_books.dtype(), ValueType, {
      int64_t num_items = compressed_tensor.size(0);
      torch::Tensor decompressed = torch::empty(
          {num_items, out_dim},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _DecompressVQKernel<IdType, ValueType, TILE_SIZE><<<grid, block>>>(
          num_items, code_books.size(0), code_books.size(1), code_books.size(2),
          out_dim, compressed_tensor.data_ptr<IdType>(),
          code_books.data_ptr<ValueType>(), decompressed.data_ptr<ValueType>());
      return decompressed;
    });
  });
  return torch::Tensor();
}

}  // namespace bifeat