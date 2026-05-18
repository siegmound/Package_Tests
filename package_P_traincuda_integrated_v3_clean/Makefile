CUDA_ARCHS ?= 80
BUILD_DIR ?= build

.PHONY: build build-a100 build-sm120 build-multi clean check

build:
	bash scripts/build_cuda_lib.sh --arch "$(CUDA_ARCHS)" --build-dir "$(BUILD_DIR)"

build-a100:
	bash scripts/build_cuda_lib.sh --arch 80 --build-dir "$(BUILD_DIR)" --clean

build-sm120:
	bash scripts/build_cuda_lib.sh --arch 120 --build-dir "$(BUILD_DIR)" --clean

build-multi:
	bash scripts/build_cuda_lib.sh --arch '80;120' --build-dir "$(BUILD_DIR)" --clean

clean:
	rm -rf "$(BUILD_DIR)"

check:
	bash scripts/check_cuda_lib_arch.sh "$(BUILD_DIR)/libpomdp_backup_cuda.so"

.PHONY: build-a100-auto build-a100-nvhpc check-a100

build-a100-auto:
	bash scripts/strategy1_install_cmake_autobuild_a100.sh

build-a100-nvhpc:
	bash -lc 'source scripts/setup_env_nvhpc_cuda13_a100.sh && bash scripts/strategy1_install_cmake_autobuild_a100.sh'

check-a100:
	bash scripts/check_runtime_deps_a100.sh
