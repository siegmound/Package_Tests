#!/usr/bin/env bash
# Source this file before building/running when using NVIDIA HPC SDK 25.11 CUDA/cuBLAS 13.
# Usage:
#   source setup_env_nvhpc_cuda13_a100.sh

export NVARCH=$(uname -s)_$(uname -m)
export NVCOMPILERS=/opt/nvidia/hpc_sdk
export NVHPC_VERSION=25.11
export NVHPC_ROOT=$NVCOMPILERS/$NVARCH/$NVHPC_VERSION

export CUDA_HOME=$NVHPC_ROOT/cuda/13.0
export CUDAToolkit_ROOT=$CUDA_HOME
export CUBLAS13_LIB=$NVHPC_ROOT/math_libs/13.0/targets/x86_64-linux/lib

export MANPATH=${MANPATH:-}:$NVHPC_ROOT/compilers/man
export PATH=$CUDA_HOME/bin:$NVHPC_ROOT/compilers/bin:$PATH
export LD_LIBRARY_PATH=$CUBLAS13_LIB:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo "[NVHPC] NVHPC_ROOT=$NVHPC_ROOT"
echo "[NVHPC] CUDA_HOME=$CUDA_HOME"
echo "[NVHPC] CUBLAS13_LIB=$CUBLAS13_LIB"
echo "[NVHPC] nvcc=$(command -v nvcc || true)"
