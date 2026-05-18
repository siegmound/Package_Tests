#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
C ABI for in-process sparse CUDA POMDP backup.

All arrays are row-major, contiguous, and owned by the caller during create/run.
The backend copies static arrays to persistent device memory at create time.

Static arrays:
  T_nnz [nA,nS] int32
  T_idx [nA,nS,max_nnz] int32
  T_val [nA,nS,max_nnz] double
  O     [nO,nA,nS] double
  R     [nA,nS] double

Dynamic arrays per backup:
  B       [nB,nS] double
  Gamma   [nG,nS] double
  BKP     [nB,nS] double output
  actions [nB] int32 output

Version hints:
  generic / k1 : direct K1 correctness backend
  v4          : sparse precompute-GAO backend
  v7          : sparse precompute-GAO backend, selected by auto for medium nG
  v8          : sparse precompute-GAO backend, selected by auto for large nG
  auto        : v4 if nG < 20, v7 if nG < 1200, v8 otherwise

K2 deliberately keeps the __global__ math independent from Python and preserves
persistent static device storage. The v4/v7/v8 hints expose the dispatch layer
needed by K3/K4; later patches can replace the precompute implementation with
the exact J kernels without changing the Python/C ABI.
*/

int pomdp_backup_create(
    int nS,
    int nA,
    int nO,
    int max_nnz,
    double gamma,
    const int* T_nnz,
    const int* T_idx,
    const double* T_val,
    const double* O,
    const double* R,
    int max_nB,
    int max_nG,
    void** out_handle
);

int pomdp_backup_run(
    void* handle,
    int nB,
    int nG,
    const double* B,
    const double* Gamma,
    double* BKP,
    int* actions,
    const char* version_hint
);

int pomdp_backup_get_last_version(
    void* handle,
    char* out_buf,
    int out_len
);

void pomdp_backup_destroy(void* handle);

#ifdef __cplusplus
}
#endif
