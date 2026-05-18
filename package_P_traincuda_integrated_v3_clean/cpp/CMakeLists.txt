cmake_minimum_required(VERSION 3.18)
project(pomdp_backup_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default target is NVIDIA A100 / Ampere (compute capability 8.0).
# Override from the command line, for example:
#   cmake -S cpp -B build -DCMAKE_CUDA_ARCHITECTURES=80
#   cmake -S cpp -B build -DCMAKE_CUDA_ARCHITECTURES="80;120"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

add_library(pomdp_backup_cuda SHARED
    src/pomdp_backup_api.cu
)

target_include_directories(pomdp_backup_cuda PUBLIC include)
target_compile_options(pomdp_backup_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(pomdp_backup_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
