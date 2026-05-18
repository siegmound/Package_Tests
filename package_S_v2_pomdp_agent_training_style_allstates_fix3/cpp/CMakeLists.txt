cmake_minimum_required(VERSION 3.18)
project(pomdp_backup_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

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
target_link_libraries(pomdp_backup_cuda PRIVATE CUDA::cublas)
target_compile_options(pomdp_backup_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(pomdp_backup_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Embed cuBLAS runtime path when CMake can resolve the imported CUDA::cublas target.
get_target_property(CUBLAS_IMPORTED_LOCATION CUDA::cublas IMPORTED_LOCATION)
if(CUBLAS_IMPORTED_LOCATION)
    get_filename_component(CUBLAS_IMPORTED_DIR "${CUBLAS_IMPORTED_LOCATION}" DIRECTORY)
    set_target_properties(pomdp_backup_cuda PROPERTIES BUILD_RPATH "${CUBLAS_IMPORTED_DIR}")
endif()
