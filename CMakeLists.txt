cmake_minimum_required(VERSION 3.15.0)
project(BIFEAT LANGUAGES CUDA CXX)

# set include
include_directories("src")


# find torch
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
set(PYTHON_INTERP python)
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
list(APPEND BIFEAT_EXTERNAL_LIB ${TORCH_LIBRARIES})


# set flag
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Define our library target
file(GLOB BIFEAT_SRC
  src/*.cu
  src/*.cc
)
add_library(bifeat SHARED ${BIFEAT_SRC})
set_target_properties(bifeat PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bifeat PROPERTIES CUDA_STANDARD 14)
set_target_properties(bifeat PROPERTIES POSITION_INDEPENDENT_CODE ON)
TARGET_COMPILE_OPTIONS(
  bifeat PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# Enable C++14
target_compile_features(bifeat PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(bifeat ${BIFEAT_EXTERNAL_LIB})
