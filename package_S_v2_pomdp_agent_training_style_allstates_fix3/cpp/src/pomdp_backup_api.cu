#include "pomdp_backup_api.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <string>

#define CHECK_CUDA(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        return -100; \
    } \
} while (0)

#define CHECK_CUBLAS(expr) do { \
    cublasStatus_t _st = (expr); \
    if (_st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, static_cast<int>(_st)); \
        return -200; \
    } \
} while (0)

struct BackupHandle {
    int nS = 0;
    int nA = 0;
    int nO = 0;
    int max_nnz = 0;
    double gamma = 0.0;
    int max_nB = 0;
    int max_nG = 0;

    char last_version[64] = "uninitialized";
    cublasHandle_t blas = nullptr;

    // Static device buffers. Allocated/copied once in create().
    int* d_T_nnz = nullptr;       // [nA,nS]
    int* d_T_idx = nullptr;       // [nA,nS,max_nnz]
    double* d_T_val = nullptr;    // [nA,nS,max_nnz]
    double* d_O = nullptr;        // [nO,nA,nS]
    double* d_R = nullptr;        // [nA,nS]

    // Dynamic persistent device buffers.
    double* d_B = nullptr;        // [max_nB,nS]
    double* d_Gamma = nullptr;    // [max_nG,nS]
    double* d_BKP = nullptr;      // [max_nB,nS]
    int* d_actions = nullptr;     // [max_nB]

    // Internal buffers for generic K1 path.
    int* d_gstar = nullptr;             // [max_nB,nA,nO]
    double* d_GAB = nullptr;            // [max_nB,nA,nS]
    double* d_action_values = nullptr;  // [max_nB,nA]

    // K2 optimized precompute path buffer.
    double* d_GAO = nullptr;            // [max_nG,nO,nA,nS]

    // Real cuBLAS v7/v8 workspaces, allocated lazily.
    double* d_GAO_col_all = nullptr;
    double* d_Y_all = nullptr;
    double* d_GAO_col_action = nullptr;
    double* d_Y_action = nullptr;
};

static void set_last_version(BackupHandle* h, const char* s) {
    if (!h) return;
    if (!s) s = "unknown";
    std::snprintf(h->last_version, sizeof(h->last_version), "%s", s);
}

static const char* normalize_hint(const char* version_hint) {
    if (!version_hint || version_hint[0] == '\0') return "auto";
    return version_hint;
}

static const char* choose_version(const char* hint, int nB, int nG) {
    hint = normalize_hint(hint);
    if (std::strcmp(hint, "auto") == 0 || std::strcmp(hint, "auto_real") == 0) {
        // Real dispatch: sparse path for small batches, global cuBLAS for medium/large
        // batches, and per-action cuBLAS for very large alpha banks.
        if (nB < 16 || nG < 128) return "v4_sparse_precompute";
        if (nG < 4096) return "v7_cublas_all";
        return "v8_cublas_by_action";
    }
    if (std::strcmp(hint, "k1") == 0 || std::strcmp(hint, "generic") == 0 || std::strcmp(hint, "generic_sparse_direct") == 0) return "generic";
    if (std::strcmp(hint, "v4") == 0 || std::strcmp(hint, "v4_sparse_precompute") == 0 || std::strcmp(hint, "sparse_precompute_current") == 0) return "v4_sparse_precompute";
    if (std::strcmp(hint, "v7") == 0 || std::strcmp(hint, "v7_cublas_all") == 0) return "v7_cublas_all";
    if (std::strcmp(hint, "v8") == 0 || std::strcmp(hint, "v8_cublas_by_action") == 0) return "v8_cublas_by_action";
    return "v4_sparse_precompute_fallback_unknown_hint";
}

__device__ __forceinline__ double sparse_gia_value(
    int a,
    int o,
    int s,
    int i,
    const int* __restrict__ T_nnz,
    const int* __restrict__ T_idx,
    const double* __restrict__ T_val,
    const double* __restrict__ O,
    const double* __restrict__ Gamma,
    int max_nnz,
    int nA,
    int nS)
{
    const long long row = static_cast<long long>(a) * nS + s;
    const long long base = row * max_nnz;
    const int count = T_nnz[row];
    double acc = 0.0;
    for (int k = 0; k < count; ++k) {
        const int sp = T_idx[base + k];
        const double tv = T_val[base + k];
        acc += tv * O[((static_cast<long long>(o) * nA + a) * nS + sp)] * Gamma[static_cast<long long>(i) * nS + sp];
    }
    return acc;
}

// -------------------- K1 generic correctness path --------------------

__global__ void kernel_select_gstar_sparse(
    int* __restrict__ gstar,
    const double* __restrict__ B,
    const double* __restrict__ Gamma,
    const int* __restrict__ T_nnz,
    const int* __restrict__ T_idx,
    const double* __restrict__ T_val,
    const double* __restrict__ O,
    int max_nnz,
    int nB,
    int nG,
    int nA,
    int nO,
    int nS)
{
    const int j = blockIdx.x;
    const int a = blockIdx.y;
    const int o = blockIdx.z;
    const int tid = threadIdx.x;

    extern __shared__ unsigned char smem[];
    double* sh_score = reinterpret_cast<double*>(smem);
    int* sh_idx = reinterpret_cast<int*>(sh_score + blockDim.x);

    double best = -DBL_MAX;
    int best_i = 0;

    for (int i = tid; i < nG; i += blockDim.x) {
        double score = 0.0;
        for (int s = 0; s < nS; ++s) {
            const double gia = sparse_gia_value(a, o, s, i, T_nnz, T_idx, T_val, O, Gamma, max_nnz, nA, nS);
            score += B[static_cast<long long>(j) * nS + s] * gia;
        }
        if (score > best || (score == best && i < best_i)) {
            best = score;
            best_i = i;
        }
    }

    sh_score[tid] = best;
    sh_idx[tid] = best_i;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double other = sh_score[tid + stride];
            const int other_i = sh_idx[tid + stride];
            if (other > sh_score[tid] || (other == sh_score[tid] && other_i < sh_idx[tid])) {
                sh_score[tid] = other;
                sh_idx[tid] = other_i;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        gstar[(static_cast<long long>(j) * nA + a) * nO + o] = sh_idx[0];
    }
}

__global__ void kernel_build_gab_sparse(
    double* __restrict__ GAB,
    double* __restrict__ action_values,
    const int* __restrict__ gstar,
    const double* __restrict__ B,
    const double* __restrict__ Gamma,
    const int* __restrict__ T_nnz,
    const int* __restrict__ T_idx,
    const double* __restrict__ T_val,
    const double* __restrict__ O,
    const double* __restrict__ R,
    double gamma_discount,
    int max_nnz,
    int nB,
    int nG,
    int nA,
    int nO,
    int nS)
{
    (void)nB;
    (void)nG;
    const int j = blockIdx.x;
    const int a = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ double sh_sum[];
    double local_value = 0.0;

    for (int s = tid; s < nS; s += blockDim.x) {
        double v = R[static_cast<long long>(a) * nS + s];
        for (int o = 0; o < nO; ++o) {
            const int i_star = gstar[(static_cast<long long>(j) * nA + a) * nO + o];
            const double gia = sparse_gia_value(a, o, s, i_star, T_nnz, T_idx, T_val, O, Gamma, max_nnz, nA, nS);
            v += gamma_discount * gia;
        }
        GAB[(static_cast<long long>(j) * nA + a) * nS + s] = v;
        local_value += B[static_cast<long long>(j) * nS + s] * v;
    }

    sh_sum[tid] = local_value;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        action_values[static_cast<long long>(j) * nA + a] = sh_sum[0];
    }
}

__global__ void kernel_select_action_copy_bkp(
    double* __restrict__ BKP,
    int* __restrict__ actions,
    const double* __restrict__ GAB,
    const double* __restrict__ action_values,
    int nB,
    int nA,
    int nS)
{
    const int j = blockIdx.x;
    const int tid = threadIdx.x;
    if (j >= nB) return;

    __shared__ int a_star;
    if (tid == 0) {
        int best_a = 0;
        double best_v = action_values[static_cast<long long>(j) * nA];
        for (int a = 1; a < nA; ++a) {
            const double v = action_values[static_cast<long long>(j) * nA + a];
            if (v > best_v) {
                best_v = v;
                best_a = a;
            }
        }
        a_star = best_a;
        actions[j] = best_a;
    }
    __syncthreads();

    for (int s = tid; s < nS; s += blockDim.x) {
        BKP[static_cast<long long>(j) * nS + s] = GAB[(static_cast<long long>(j) * nA + a_star) * nS + s];
    }
}

// -------------------- K2 sparse precompute-GAO path --------------------

__global__ void kernel_build_GAO_sparse_precompute(
    double* __restrict__ GAO,
    const double* __restrict__ Gamma,
    const int* __restrict__ T_nnz,
    const int* __restrict__ T_idx,
    const double* __restrict__ T_val,
    const double* __restrict__ O,
    int max_nnz,
    int nG,
    int nO,
    int nA,
    int nS)
{
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(nG) * nO * nA * nS;
    if (idx >= total) return;

    int rem = static_cast<int>(idx);
    const int s = rem % nS; rem /= nS;
    const int a = rem % nA; rem /= nA;
    const int o = rem % nO; rem /= nO;
    const int i = rem;

    GAO[idx] = sparse_gia_value(a, o, s, i, T_nnz, T_idx, T_val, O, Gamma, max_nnz, nA, nS);
}

__global__ void kernel_select_gstar_from_GAO(
    int* __restrict__ gstar,
    const double* __restrict__ B,
    const double* __restrict__ GAO,
    int nB,
    int nG,
    int nA,
    int nO,
    int nS)
{
    const int j = blockIdx.x;
    const int a = blockIdx.y;
    const int o = blockIdx.z;
    const int tid = threadIdx.x;

    extern __shared__ unsigned char smem[];
    double* sh_score = reinterpret_cast<double*>(smem);
    int* sh_idx = reinterpret_cast<int*>(sh_score + blockDim.x);

    double best = -DBL_MAX;
    int best_i = 0;

    for (int i = tid; i < nG; i += blockDim.x) {
        double score = 0.0;
        const long long base = (((static_cast<long long>(i) * nO + o) * nA + a) * nS);
        const long long bbase = static_cast<long long>(j) * nS;
        for (int s = 0; s < nS; ++s) {
            score += B[bbase + s] * GAO[base + s];
        }
        if (score > best || (score == best && i < best_i)) {
            best = score;
            best_i = i;
        }
    }

    sh_score[tid] = best;
    sh_idx[tid] = best_i;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double other = sh_score[tid + stride];
            const int other_i = sh_idx[tid + stride];
            if (other > sh_score[tid] || (other == sh_score[tid] && other_i < sh_idx[tid])) {
                sh_score[tid] = other;
                sh_idx[tid] = other_i;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        gstar[(static_cast<long long>(j) * nA + a) * nO + o] = sh_idx[0];
    }
}

__global__ void kernel_build_gab_from_GAO(
    double* __restrict__ GAB,
    double* __restrict__ action_values,
    const int* __restrict__ gstar,
    const double* __restrict__ B,
    const double* __restrict__ GAO,
    const double* __restrict__ R,
    double gamma_discount,
    int nB,
    int nG,
    int nA,
    int nO,
    int nS)
{
    (void)nB;
    (void)nG;
    const int j = blockIdx.x;
    const int a = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ double sh_sum[];
    double local_value = 0.0;

    for (int s = tid; s < nS; s += blockDim.x) {
        double v = R[static_cast<long long>(a) * nS + s];
        for (int o = 0; o < nO; ++o) {
            const int i_star = gstar[(static_cast<long long>(j) * nA + a) * nO + o];
            const long long idx = (((static_cast<long long>(i_star) * nO + o) * nA + a) * nS + s);
            v += gamma_discount * GAO[idx];
        }
        GAB[(static_cast<long long>(j) * nA + a) * nS + s] = v;
        local_value += B[static_cast<long long>(j) * nS + s] * v;
    }

    sh_sum[tid] = local_value;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        action_values[static_cast<long long>(j) * nA + a] = sh_sum[0];
    }
}


// -------------------- Real cuBLAS v7/v8 selection paths --------------------
// Pack the already-computed sparse GAO tensor into column-major matrices for cuBLAS.
// B is row-major [nB,nS], which is layout-compatible with column-major [nS,nB].

__global__ void kernel_pack_GAO_col_all_from_GAO(
    double* __restrict__ A_col,
    const double* __restrict__ GAO,
    int nG,
    int nO,
    int nA,
    int nS)
{
    const long long total = static_cast<long long>(nA) * nO * nG * nS;
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int s = static_cast<int>(t % nS);
    long long q = t / nS;
    const int i = static_cast<int>(q % nG); q /= nG;
    const int o = static_cast<int>(q % nO); q /= nO;
    const int a = static_cast<int>(q);
    const long long K = static_cast<long long>(nA) * nO * nG;
    const long long row = (static_cast<long long>(a) * nO + o) * nG + i;
    const long long gidx = (((static_cast<long long>(i) * nO + o) * nA + a) * nS + s);
    A_col[row + K * static_cast<long long>(s)] = GAO[gidx];
}

__global__ void kernel_argmax_from_Y_all(
    int* __restrict__ gstar,
    const double* __restrict__ Y,
    int nB,
    int nG,
    int nA,
    int nO)
{
    const int j = blockIdx.x;
    const int a = blockIdx.y;
    const int o = blockIdx.z;
    if (j >= nB || a >= nA || o >= nO) return;
    const long long K = static_cast<long long>(nA) * nO * nG;
    const long long base = (static_cast<long long>(a) * nO + o) * nG;
    const long long ycol = K * static_cast<long long>(j);
    double best = -DBL_MAX;
    int best_i = 0;
    for (int i = 0; i < nG; ++i) {
        const double v = Y[ycol + base + i];
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    gstar[(static_cast<long long>(j) * nA + a) * nO + o] = best_i;
}

__global__ void kernel_pack_GAO_col_action_from_GAO(
    double* __restrict__ A_col,
    const double* __restrict__ GAO,
    int action,
    int nG,
    int nO,
    int nA,
    int nS)
{
    const long long total = static_cast<long long>(nO) * nG * nS;
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int s = static_cast<int>(t % nS);
    long long q = t / nS;
    const int i = static_cast<int>(q % nG); q /= nG;
    const int o = static_cast<int>(q);
    const long long K = static_cast<long long>(nO) * nG;
    const long long row = static_cast<long long>(o) * nG + i;
    const long long gidx = (((static_cast<long long>(i) * nO + o) * nA + action) * nS + s);
    A_col[row + K * static_cast<long long>(s)] = GAO[gidx];
}

__global__ void kernel_argmax_from_Y_action(
    int* __restrict__ gstar,
    const double* __restrict__ Y,
    int action,
    int nB,
    int nG,
    int nA,
    int nO)
{
    const int j = blockIdx.x;
    const int o = blockIdx.y;
    if (j >= nB || o >= nO) return;
    const long long K = static_cast<long long>(nO) * nG;
    const long long base = static_cast<long long>(o) * nG;
    const long long ycol = K * static_cast<long long>(j);
    double best = -DBL_MAX;
    int best_i = 0;
    for (int i = 0; i < nG; ++i) {
        const double v = Y[ycol + base + i];
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    gstar[(static_cast<long long>(j) * nA + action) * nO + o] = best_i;
}

static int free_dynamic_buffers(BackupHandle* h) {
    if (!h) return 0;
    if (h->d_B) cudaFree(h->d_B);
    if (h->d_Gamma) cudaFree(h->d_Gamma);
    if (h->d_BKP) cudaFree(h->d_BKP);
    if (h->d_actions) cudaFree(h->d_actions);
    if (h->d_gstar) cudaFree(h->d_gstar);
    if (h->d_GAB) cudaFree(h->d_GAB);
    if (h->d_action_values) cudaFree(h->d_action_values);
    if (h->d_GAO) cudaFree(h->d_GAO);
    if (h->d_GAO_col_all) cudaFree(h->d_GAO_col_all);
    if (h->d_Y_all) cudaFree(h->d_Y_all);
    if (h->d_GAO_col_action) cudaFree(h->d_GAO_col_action);
    if (h->d_Y_action) cudaFree(h->d_Y_action);
    h->d_B = nullptr;
    h->d_Gamma = nullptr;
    h->d_BKP = nullptr;
    h->d_actions = nullptr;
    h->d_gstar = nullptr;
    h->d_GAB = nullptr;
    h->d_action_values = nullptr;
    h->d_GAO = nullptr;
    h->d_GAO_col_all = nullptr;
    h->d_Y_all = nullptr;
    h->d_GAO_col_action = nullptr;
    h->d_Y_action = nullptr;
    h->max_nB = 0;
    h->max_nG = 0;
    return 0;
}

static int ensure_dynamic_buffers(BackupHandle* h, int nB, int nG) {
    if (nB <= h->max_nB && nG <= h->max_nG && h->d_B && h->d_Gamma && h->d_BKP && h->d_actions && h->d_gstar && h->d_GAB && h->d_action_values && h->d_GAO) {
        return 0;
    }
    free_dynamic_buffers(h);
    h->max_nB = nB;
    h->max_nG = nG;
    CHECK_CUDA(cudaMalloc(&h->d_B, sizeof(double) * (size_t)nB * h->nS));
    CHECK_CUDA(cudaMalloc(&h->d_Gamma, sizeof(double) * (size_t)nG * h->nS));
    CHECK_CUDA(cudaMalloc(&h->d_BKP, sizeof(double) * (size_t)nB * h->nS));
    CHECK_CUDA(cudaMalloc(&h->d_actions, sizeof(int) * (size_t)nB));
    CHECK_CUDA(cudaMalloc(&h->d_gstar, sizeof(int) * (size_t)nB * h->nA * h->nO));
    CHECK_CUDA(cudaMalloc(&h->d_GAB, sizeof(double) * (size_t)nB * h->nA * h->nS));
    CHECK_CUDA(cudaMalloc(&h->d_action_values, sizeof(double) * (size_t)nB * h->nA));
    CHECK_CUDA(cudaMalloc(&h->d_GAO, sizeof(double) * (size_t)nG * h->nO * h->nA * h->nS));
    return 0;
}


static int ensure_v7_workspace(BackupHandle* h, int nB, int nG) {
    const int cap_nB = (h->max_nB > nB) ? h->max_nB : nB;
    const int cap_nG = (h->max_nG > nG) ? h->max_nG : nG;
    const size_t K_all = static_cast<size_t>(h->nA) * h->nO * cap_nG;
    if (!h->d_GAO_col_all) CHECK_CUDA(cudaMalloc(&h->d_GAO_col_all, sizeof(double) * K_all * h->nS));
    if (!h->d_Y_all) CHECK_CUDA(cudaMalloc(&h->d_Y_all, sizeof(double) * K_all * cap_nB));
    return 0;
}

static int ensure_v8_workspace(BackupHandle* h, int nB, int nG) {
    const int cap_nB = (h->max_nB > nB) ? h->max_nB : nB;
    const int cap_nG = (h->max_nG > nG) ? h->max_nG : nG;
    const size_t K_action = static_cast<size_t>(h->nO) * cap_nG;
    if (!h->d_GAO_col_action) CHECK_CUDA(cudaMalloc(&h->d_GAO_col_action, sizeof(double) * K_action * h->nS));
    if (!h->d_Y_action) CHECK_CUDA(cudaMalloc(&h->d_Y_action, sizeof(double) * K_action * cap_nB));
    return 0;
}

static int launch_sparse_backup_generic(BackupHandle* h, int nB, int nG) {
    if (!h || nB <= 0 || nG <= 0) return -2;

    const int threads = 256;
    const dim3 grid_select(nB, h->nA, h->nO);
    const size_t smem_select = threads * (sizeof(double) + sizeof(int));
    kernel_select_gstar_sparse<<<grid_select, threads, smem_select>>>(
        h->d_gstar,
        h->d_B,
        h->d_Gamma,
        h->d_T_nnz,
        h->d_T_idx,
        h->d_T_val,
        h->d_O,
        h->max_nnz,
        nB,
        nG,
        h->nA,
        h->nO,
        h->nS);
    CHECK_CUDA(cudaGetLastError());

    const dim3 grid_gab(nB, h->nA);
    const size_t smem_gab = threads * sizeof(double);
    kernel_build_gab_sparse<<<grid_gab, threads, smem_gab>>>(
        h->d_GAB,
        h->d_action_values,
        h->d_gstar,
        h->d_B,
        h->d_Gamma,
        h->d_T_nnz,
        h->d_T_idx,
        h->d_T_val,
        h->d_O,
        h->d_R,
        h->gamma,
        h->max_nnz,
        nB,
        nG,
        h->nA,
        h->nO,
        h->nS);
    CHECK_CUDA(cudaGetLastError());

    const int copy_threads = 256;
    kernel_select_action_copy_bkp<<<nB, copy_threads>>>(
        h->d_BKP,
        h->d_actions,
        h->d_GAB,
        h->d_action_values,
        nB,
        h->nA,
        h->nS);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

static int launch_sparse_backup_precompute(BackupHandle* h, int nB, int nG, const char* selected_version) {
    if (!h || nB <= 0 || nG <= 0) return -2;
    (void)selected_version;

    const int threads = 256;
    const long long total_GAO = static_cast<long long>(nG) * h->nO * h->nA * h->nS;
    const int blocks_GAO = static_cast<int>((total_GAO + threads - 1) / threads);
    kernel_build_GAO_sparse_precompute<<<blocks_GAO, threads>>>(
        h->d_GAO,
        h->d_Gamma,
        h->d_T_nnz,
        h->d_T_idx,
        h->d_T_val,
        h->d_O,
        h->max_nnz,
        nG,
        h->nO,
        h->nA,
        h->nS);
    CHECK_CUDA(cudaGetLastError());

    const dim3 grid_select(nB, h->nA, h->nO);
    const size_t smem_select = threads * (sizeof(double) + sizeof(int));
    kernel_select_gstar_from_GAO<<<grid_select, threads, smem_select>>>(
        h->d_gstar,
        h->d_B,
        h->d_GAO,
        nB,
        nG,
        h->nA,
        h->nO,
        h->nS);
    CHECK_CUDA(cudaGetLastError());

    const dim3 grid_gab(nB, h->nA);
    const size_t smem_gab = threads * sizeof(double);
    kernel_build_gab_from_GAO<<<grid_gab, threads, smem_gab>>>(
        h->d_GAB,
        h->d_action_values,
        h->d_gstar,
        h->d_B,
        h->d_GAO,
        h->d_R,
        h->gamma,
        nB,
        nG,
        h->nA,
        h->nO,
        h->nS);
    CHECK_CUDA(cudaGetLastError());

    kernel_select_action_copy_bkp<<<nB, threads>>>(
        h->d_BKP,
        h->d_actions,
        h->d_GAB,
        h->d_action_values,
        nB,
        h->nA,
        h->nS);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}


static int launch_v7_cublas_all(BackupHandle* h, int nB, int nG) {
    if (!h || nB <= 0 || nG <= 0) return -2;
    int rc = ensure_v7_workspace(h, nB, nG);
    if (rc != 0) return rc;
    const int threads = 256;
    const long long total_GAO = static_cast<long long>(nG) * h->nO * h->nA * h->nS;
    const int blocks_GAO = static_cast<int>((total_GAO + threads - 1) / threads);
    kernel_build_GAO_sparse_precompute<<<blocks_GAO, threads>>>(
        h->d_GAO,
        h->d_Gamma,
        h->d_T_nnz,
        h->d_T_idx,
        h->d_T_val,
        h->d_O,
        h->max_nnz,
        nG,
        h->nO,
        h->nA,
        h->nS);
    CHECK_CUDA(cudaGetLastError());

    const long long total_pack = static_cast<long long>(h->nA) * h->nO * nG * h->nS;
    const int blocks_pack = static_cast<int>((total_pack + threads - 1) / threads);
    kernel_pack_GAO_col_all_from_GAO<<<blocks_pack, threads>>>(h->d_GAO_col_all, h->d_GAO, nG, h->nO, h->nA, h->nS);
    CHECK_CUDA(cudaGetLastError());

    const int K_all = h->nA * h->nO * nG;
    const double alpha = 1.0;
    const double beta = 0.0;
    CHECK_CUBLAS(cublasDgemm(
        h->blas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        K_all,
        nB,
        h->nS,
        &alpha,
        h->d_GAO_col_all,
        K_all,
        h->d_B,
        h->nS,
        &beta,
        h->d_Y_all,
        K_all));

    const dim3 grid_arg(nB, h->nA, h->nO);
    kernel_argmax_from_Y_all<<<grid_arg, 1>>>(h->d_gstar, h->d_Y_all, nB, nG, h->nA, h->nO);
    CHECK_CUDA(cudaGetLastError());

    const dim3 grid_gab(nB, h->nA);
    const size_t smem_gab = threads * sizeof(double);
    kernel_build_gab_from_GAO<<<grid_gab, threads, smem_gab>>>(
        h->d_GAB, h->d_action_values, h->d_gstar, h->d_B, h->d_GAO, h->d_R,
        h->gamma, nB, nG, h->nA, h->nO, h->nS);
    CHECK_CUDA(cudaGetLastError());
    kernel_select_action_copy_bkp<<<nB, threads>>>(h->d_BKP, h->d_actions, h->d_GAB, h->d_action_values, nB, h->nA, h->nS);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

static int launch_v8_cublas_by_action(BackupHandle* h, int nB, int nG) {
    if (!h || nB <= 0 || nG <= 0) return -2;
    int rc = ensure_v8_workspace(h, nB, nG);
    if (rc != 0) return rc;
    const int threads = 256;
    const long long total_GAO = static_cast<long long>(nG) * h->nO * h->nA * h->nS;
    const int blocks_GAO = static_cast<int>((total_GAO + threads - 1) / threads);
    kernel_build_GAO_sparse_precompute<<<blocks_GAO, threads>>>(
        h->d_GAO,
        h->d_Gamma,
        h->d_T_nnz,
        h->d_T_idx,
        h->d_T_val,
        h->d_O,
        h->max_nnz,
        nG,
        h->nO,
        h->nA,
        h->nS);
    CHECK_CUDA(cudaGetLastError());

    const double alpha = 1.0;
    const double beta = 0.0;
    const int K_action = h->nO * nG;
    const long long total_pack = static_cast<long long>(h->nO) * nG * h->nS;
    const int blocks_pack = static_cast<int>((total_pack + threads - 1) / threads);
    for (int a = 0; a < h->nA; ++a) {
        kernel_pack_GAO_col_action_from_GAO<<<blocks_pack, threads>>>(h->d_GAO_col_action, h->d_GAO, a, nG, h->nO, h->nA, h->nS);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUBLAS(cublasDgemm(
            h->blas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            K_action,
            nB,
            h->nS,
            &alpha,
            h->d_GAO_col_action,
            K_action,
            h->d_B,
            h->nS,
            &beta,
            h->d_Y_action,
            K_action));
        const dim3 grid_arg(nB, h->nO);
        kernel_argmax_from_Y_action<<<grid_arg, 1>>>(h->d_gstar, h->d_Y_action, a, nB, nG, h->nA, h->nO);
        CHECK_CUDA(cudaGetLastError());
    }

    const dim3 grid_gab(nB, h->nA);
    const size_t smem_gab = threads * sizeof(double);
    kernel_build_gab_from_GAO<<<grid_gab, threads, smem_gab>>>(
        h->d_GAB, h->d_action_values, h->d_gstar, h->d_B, h->d_GAO, h->d_R,
        h->gamma, nB, nG, h->nA, h->nO, h->nS);
    CHECK_CUDA(cudaGetLastError());
    kernel_select_action_copy_bkp<<<nB, threads>>>(h->d_BKP, h->d_actions, h->d_GAB, h->d_action_values, nB, h->nA, h->nS);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

static int launch_sparse_backup(BackupHandle* h, int nB, int nG, const char* version_hint) {
    const char* selected = choose_version(version_hint, nB, nG);
    if (std::strcmp(selected, "generic") == 0) {
        set_last_version(h, "generic_sparse_direct");
        return launch_sparse_backup_generic(h, nB, nG);
    }
    if (std::strcmp(selected, "v4_sparse_precompute") == 0) {
        set_last_version(h, "v4_sparse_precompute");
        return launch_sparse_backup_precompute(h, nB, nG, "v4_sparse_precompute");
    }
    if (std::strcmp(selected, "v7_cublas_all") == 0) {
        set_last_version(h, "v7_cublas_all");
        return launch_v7_cublas_all(h, nB, nG);
    }
    if (std::strcmp(selected, "v8_cublas_by_action") == 0) {
        set_last_version(h, "v8_cublas_by_action");
        return launch_v8_cublas_by_action(h, nB, nG);
    }
    set_last_version(h, "v4_sparse_precompute_fallback_unknown_hint");
    return launch_sparse_backup_precompute(h, nB, nG, "v4_sparse_precompute");
}

extern "C" int pomdp_backup_create(
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
) {
    if (!out_handle || !T_nnz || !T_idx || !T_val || !O || !R) return -1;
    if (nS <= 0 || nA <= 0 || nO <= 0 || max_nnz <= 0) return -2;
    BackupHandle* h = new BackupHandle();
    h->nS = nS; h->nA = nA; h->nO = nO; h->max_nnz = max_nnz; h->gamma = gamma;
    set_last_version(h, "created");
    CHECK_CUBLAS(cublasCreate(&h->blas));
    size_t T_nnz_bytes = sizeof(int) * (size_t)nA * nS;
    size_t T_ell_int_bytes = sizeof(int) * (size_t)nA * nS * max_nnz;
    size_t T_ell_val_bytes = sizeof(double) * (size_t)nA * nS * max_nnz;
    size_t O_bytes = sizeof(double) * (size_t)nO * nA * nS;
    size_t R_bytes = sizeof(double) * (size_t)nA * nS;
    CHECK_CUDA(cudaMalloc(&h->d_T_nnz, T_nnz_bytes));
    CHECK_CUDA(cudaMalloc(&h->d_T_idx, T_ell_int_bytes));
    CHECK_CUDA(cudaMalloc(&h->d_T_val, T_ell_val_bytes));
    CHECK_CUDA(cudaMalloc(&h->d_O, O_bytes));
    CHECK_CUDA(cudaMalloc(&h->d_R, R_bytes));
    CHECK_CUDA(cudaMemcpy(h->d_T_nnz, T_nnz, T_nnz_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_T_idx, T_idx, T_ell_int_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_T_val, T_val, T_ell_val_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_O, O, O_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_R, R, R_bytes, cudaMemcpyHostToDevice));
    if (max_nB > 0 && max_nG > 0) {
        int rc = ensure_dynamic_buffers(h, max_nB, max_nG);
        if (rc != 0) {
            pomdp_backup_destroy((void*)h);
            return rc;
        }
    }
    *out_handle = (void*)h;
    return 0;
}

extern "C" int pomdp_backup_run(
    void* handle,
    int nB,
    int nG,
    const double* B,
    const double* Gamma,
    double* BKP,
    int* actions,
    const char* version_hint
) {
    if (!handle || !B || !Gamma || !BKP || !actions) return -1;
    BackupHandle* h = (BackupHandle*)handle;
    if (nB <= 0 || nG <= 0) return -2;
    int rc = ensure_dynamic_buffers(h, nB, nG);
    if (rc != 0) return rc;
    CHECK_CUDA(cudaMemcpy(h->d_B, B, sizeof(double) * (size_t)nB * h->nS, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_Gamma, Gamma, sizeof(double) * (size_t)nG * h->nS, cudaMemcpyHostToDevice));
    rc = launch_sparse_backup(h, nB, nG, version_hint ? version_hint : "auto");
    if (rc != 0) return rc;
    CHECK_CUDA(cudaMemcpy(BKP, h->d_BKP, sizeof(double) * (size_t)nB * h->nS, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(actions, h->d_actions, sizeof(int) * (size_t)nB, cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int pomdp_backup_get_last_version(void* handle, char* out_buf, int out_len) {
    if (!handle || !out_buf || out_len <= 0) return -1;
    BackupHandle* h = (BackupHandle*)handle;
    std::snprintf(out_buf, (size_t)out_len, "%s", h->last_version);
    return 0;
}

extern "C" void pomdp_backup_destroy(void* handle) {
    if (!handle) return;
    BackupHandle* h = (BackupHandle*)handle;
    if (h->d_T_nnz) cudaFree(h->d_T_nnz);
    if (h->d_T_idx) cudaFree(h->d_T_idx);
    if (h->d_T_val) cudaFree(h->d_T_val);
    if (h->d_O) cudaFree(h->d_O);
    if (h->d_R) cudaFree(h->d_R);
    if (h->blas) cublasDestroy(h->blas);
    free_dynamic_buffers(h);
    delete h;
}
