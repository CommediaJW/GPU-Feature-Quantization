#ifndef BIFEAT_CUDA_COMMON_H_
#define BIFEAT_CUDA_COMMON_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CUDA_CALL(call)                                                  \
  {                                                                      \
    cudaError_t cudaStatus = call;                                       \
    if (cudaSuccess != cudaStatus) {                                     \
      fprintf(stderr,                                                    \
              "%s:%d ERROR: CUDA RT call \"%s\" failed "                 \
              "with "                                                    \
              "%s (%d).\n",                                              \
              __FILE__, __LINE__, #call, cudaGetErrorString(cudaStatus), \
              cudaStatus);                                               \
      exit(cudaStatus);                                                  \
    }                                                                    \
  }

#define BIFEAT_INT_TYPE_SWITCH(TorchIntType, IntType, ...)               \
  do {                                                                   \
    if ((TorchIntType) == torch::kUInt8) {                               \
      typedef u_int8_t IntType;                                          \
      { __VA_ARGS__ }                                                    \
    } else if ((TorchIntType) == torch::kInt8) {                         \
      typedef int8_t IntType;                                            \
      { __VA_ARGS__ }                                                    \
    } else if ((TorchIntType) == torch::kInt16) {                        \
      typedef int16_t IntType;                                           \
      { __VA_ARGS__ }                                                    \
    } else if ((TorchIntType) == torch::kInt32) {                        \
      typedef int32_t IntType;                                           \
      { __VA_ARGS__ }                                                    \
    } else if ((TorchIntType) == torch::kInt64) {                        \
      typedef int64_t IntType;                                           \
      { __VA_ARGS__ }                                                    \
    } else {                                                             \
      LOG(FATAL)                                                         \
          << "Int can only be uint8 or int8 or int16 or int32 or int64"; \
    }                                                                    \
  } while (0);

#endif