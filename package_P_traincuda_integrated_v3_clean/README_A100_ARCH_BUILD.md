# A100 / architecture build

For A100 use compute capability 8.0:

```bash
cd package_P_traincuda_integrated_v3_1_clean
bash scripts/31_build_backend_lib.sh --arch 80 --clean
```

Other examples:

```bash
bash scripts/31_build_backend_lib.sh --arch native --clean
bash scripts/31_build_backend_lib.sh --arch 120 --clean
bash scripts/31_build_backend_lib.sh --arch '80;120' --clean
```

Use `scripts/check_cuda_lib_arch.sh build/libpomdp_backup_cuda.so` to inspect embedded code objects when `cuobjdump` is available.
