#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "pomdp_benchmark_common.hpp"

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        const cudaError_t error__ = (call);                                           \
        if (error__ != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                      << " -> " << cudaGetErrorString(error__) << std::endl;          \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

using namespace pomdpinput;
using namespace pomdpbench;

namespace {

// CUDA kernels

static constexpr int BLOCK_V4 = 256;
static constexpr int I_TILE_V4 = 8;

__global__ void kernel_build_GAO_row(
    double* GAO,
    const double* Gamma,
    const double* O,
    const double* T,
    int nG, int nO, int nA, int nS)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(nG) * nO * nA * nS;
    if (idx >= total) return;

    int s = static_cast<int>(idx % nS);
    long long tmp = idx / nS;
    int a = static_cast<int>(tmp % nA);
    tmp /= nA;
    int o = static_cast<int>(tmp % nO);
    int i = static_cast<int>(tmp / nO);

    double acc = 0.0;
    for (int sp = 0; sp < nS; ++sp) {
        acc += T[((static_cast<long long>(a) * nS + s) * nS + sp)] *
               O[((static_cast<long long>(o) * nA + a) * nS + sp)] *
               Gamma[static_cast<long long>(i) * nS + sp];
    }
    GAO[idx] = acc;
}

__inline__ __device__ double warp_reduce_sum_v4(double v)
{
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__global__ void kernel_selection_v4_from_GAO(
    int* g_star_ao,
    const double* B,
    const double* GAO,
    int nB, int nA, int nO, int nS, int nG)
{
    int j = blockIdx.x;
    int a = blockIdx.y;
    int o = blockIdx.z;
    if (j >= nB || a >= nA || o >= nO) return;

    const double* B_ptr = &B[(long long)j * nS];
    __shared__ double warp_sums[8];

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    double best_val = -1.0e300;
    int best_i = 0;

    for (int i0 = 0; i0 < nG; i0 += I_TILE_V4) {
        double acc[I_TILE_V4];
        #pragma unroll
        for (int k = 0; k < I_TILE_V4; ++k) acc[k] = 0.0;

        for (int s = threadIdx.x; s < nS; s += BLOCK_V4) {
            const double b = B_ptr[s];
            #pragma unroll
            for (int k = 0; k < I_TILE_V4; ++k) {
                const int i = i0 + k;
                if (i < nG) {
                    acc[k] += b * GAO[(((static_cast<long long>(i) * nO) + o) * nA + a) * nS + s];
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < I_TILE_V4; ++k) {
            const int i = i0 + k;
            if (i >= nG) break;
            double sum = warp_reduce_sum_v4(acc[k]);
            if (lane == 0) warp_sums[wid] = sum;
            __syncthreads();
            if (wid == 0) {
                double v = (lane < 8) ? warp_sums[lane] : 0.0;
                v = warp_reduce_sum_v4(v);
                if (lane == 0 && v > best_val) {
                    best_val = v;
                    best_i = i;
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        g_star_ao[(static_cast<long long>(j) * nA + a) * nO + o] = best_i;
    }
}

__global__ void kernel_generation_general(
    double* BKP,
    double* V_new,
    const int* g_star_ao,
    const double* B,
    const double* Gamma,
    const double* O,
    const double* T,
    const double* R,
    double gamma,
    int nB,
    int nA,
    int nO,
    int nS)
{
    int j = blockIdx.x;
    if (j >= nB) return;

    extern __shared__ double sh[];
    __shared__ int best_a;

    for (int a = 0; a < nA; ++a) {
        double local = 0.0;
        for (int s = threadIdx.x; s < nS; s += blockDim.x) {
            double so = 0.0;
            for (int o = 0; o < nO; ++o) {
                const int bi = g_star_ao[(static_cast<long long>(j) * nA + a) * nO + o];
                double gia = 0.0;
                for (int sp = 0; sp < nS; ++sp) {
                    gia += T[((static_cast<long long>(a) * nS + s) * nS + sp)] *
                           O[((static_cast<long long>(o) * nA + a) * nS + sp)] *
                           Gamma[static_cast<long long>(bi) * nS + sp];
                }
                so += gia;
            }
            local += B[static_cast<long long>(j) * nS + s] * (R[static_cast<long long>(a) * nS + s] + gamma * so);
        }

        sh[threadIdx.x] = local;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < static_cast<int>(stride)) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0) sh[blockDim.x + a] = sh[0];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double mv = -1.0e300;
        int win = 0;
        for (int a = 0; a < nA; ++a) {
            const double v = sh[blockDim.x + a];
            if (v > mv) {
                mv = v;
                win = a;
            }
        }
        best_a = win;
        V_new[j] = mv;
    }
    __syncthreads();

    const int a_sel = best_a;
    for (int s = threadIdx.x; s < nS; s += blockDim.x) {
        double so = 0.0;
        for (int o = 0; o < nO; ++o) {
            const int bi = g_star_ao[(static_cast<long long>(j) * nA + a_sel) * nO + o];
            double gia = 0.0;
            for (int sp = 0; sp < nS; ++sp) {
                gia += T[((static_cast<long long>(a_sel) * nS + s) * nS + sp)] *
                       O[((static_cast<long long>(o) * nA + a_sel) * nS + sp)] *
                       Gamma[static_cast<long long>(bi) * nS + sp];
            }
            so += gia;
        }
        BKP[static_cast<long long>(j) * nS + s] = R[static_cast<long long>(a_sel) * nS + s] + gamma * so;
    }
}

__global__ void kernel_residual(double* d_res, const double* Vn, const double* Vo, int nB)
{
    extern __shared__ double sh[];
    const int tid = threadIdx.x;
    double local_max = 0.0;
    for (int j = tid; j < nB; j += blockDim.x) local_max = fmax(local_max, fabs(Vn[j] - Vo[j]));
    sh[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < static_cast<int>(stride)) sh[tid] = fmax(sh[tid], sh[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) *d_res = sh[0];
}

// GPU execution context

struct GpuContextV4 {
    const Problem& p;
    const int nG0;
    const int nGmax;

    double* d_B = nullptr;
    double* d_T = nullptr;
    double* d_O = nullptr;
    double* d_R = nullptr;
    double* d_GammaA = nullptr;
    double* d_GammaB = nullptr;
    double* d_VoldA = nullptr;
    double* d_VoldB = nullptr;
    double* d_GAO = nullptr;
    double* d_res = nullptr;
    int* d_gstar = nullptr;

    double* d_GammaCur = nullptr;
    double* d_GammaNext = nullptr;
    double* d_VoldCur = nullptr;
    double* d_VnewNext = nullptr;
    int nG_cur = 0;

    explicit GpuContextV4(const Problem& prob)
        : p(prob),
          nG0(static_cast<int>(prob.Gamma.size() / static_cast<size_t>(prob.nS))),
          nGmax(std::max(nG0, prob.nB))
    {
        CHECK_CUDA(cudaMalloc(&d_B, static_cast<size_t>(p.nB) * p.nS * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_T, static_cast<size_t>(p.nA) * p.nS * p.nS * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_O, static_cast<size_t>(p.nO) * p.nA * p.nS * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_R, static_cast<size_t>(p.nA) * p.nS * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_GammaA, static_cast<size_t>(nGmax) * p.nS * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_GammaB, static_cast<size_t>(nGmax) * p.nS * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_VoldA, static_cast<size_t>(p.nB) * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_VoldB, static_cast<size_t>(p.nB) * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_GAO, static_cast<size_t>(nGmax) * p.nO * p.nA * p.nS * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_res, sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_gstar, static_cast<size_t>(p.nB) * p.nA * p.nO * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_B, p.B.data(), static_cast<size_t>(p.nB) * p.nS * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_T, p.T.data(), static_cast<size_t>(p.nA) * p.nS * p.nS * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_O, p.O.data(), static_cast<size_t>(p.nO) * p.nA * p.nS * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_R, p.R.data(), static_cast<size_t>(p.nA) * p.nS * sizeof(double), cudaMemcpyHostToDevice));
    }

    ~GpuContextV4()
    {
        cudaFree(d_B);
        cudaFree(d_T);
        cudaFree(d_O);
        cudaFree(d_R);
        cudaFree(d_GammaA);
        cudaFree(d_GammaB);
        cudaFree(d_VoldA);
        cudaFree(d_VoldB);
        cudaFree(d_GAO);
        cudaFree(d_res);
        cudaFree(d_gstar);
    }

    void reset(const std::vector<double>& gamma0, const std::vector<double>& vold0)
    {
        CHECK_CUDA(cudaMemcpy(d_GammaA, gamma0.data(), gamma0.size() * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_VoldA, vold0.data(), vold0.size() * sizeof(double), cudaMemcpyHostToDevice));
        d_GammaCur = d_GammaA;
        d_GammaNext = d_GammaB;
        d_VoldCur = d_VoldA;
        d_VnewNext = d_VoldB;
        nG_cur = nG0;
    }

    void iterate()
    {
        const int threads = 256;
        const size_t smem_gen = static_cast<size_t>(threads + p.nA) * sizeof(double);
        const size_t smem_res = static_cast<size_t>(threads) * sizeof(double);
        const long long totalGAO = static_cast<long long>(nG_cur) * p.nO * p.nA * p.nS;

        kernel_build_GAO_row<<<static_cast<int>((totalGAO + threads - 1) / threads), threads>>>(
            d_GAO, d_GammaCur, d_O, d_T, nG_cur, p.nO, p.nA, p.nS);
        CHECK_CUDA(cudaGetLastError());

        kernel_selection_v4_from_GAO<<<dim3(p.nB, p.nA, p.nO), threads>>>(
            d_gstar, d_B, d_GAO, p.nB, p.nA, p.nO, p.nS, nG_cur);
        CHECK_CUDA(cudaGetLastError());

        kernel_generation_general<<<p.nB, threads, smem_gen>>>(
            d_GammaNext, d_VnewNext, d_gstar, d_B, d_GammaCur, d_O, d_T, d_R,
            p.gamma, p.nB, p.nA, p.nO, p.nS);
        CHECK_CUDA(cudaGetLastError());

        kernel_residual<<<1, threads, smem_res>>>(d_res, d_VnewNext, d_VoldCur, p.nB);
        CHECK_CUDA(cudaGetLastError());

        std::swap(d_GammaCur, d_GammaNext);
        std::swap(d_VoldCur, d_VnewNext);
        nG_cur = p.nB;
    }

    double fetch_residual() const
    {
        double r = 0.0;
        CHECK_CUDA(cudaMemcpy(&r, d_res, sizeof(double), cudaMemcpyDeviceToHost));
        return r;
    }

    void dump_final(const std::string& prefix) const
    {
        std::vector<int> gstar(static_cast<size_t>(p.nB) * p.nA * p.nO);
        std::vector<double> bkp(static_cast<size_t>(p.nB) * p.nS);
        std::vector<double> vnew(static_cast<size_t>(p.nB));
        CHECK_CUDA(cudaMemcpy(gstar.data(), d_gstar, gstar.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(bkp.data(), d_GammaCur, bkp.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(vnew.data(), d_VoldCur, vnew.size() * sizeof(double), cudaMemcpyDeviceToHost));
        write_tensor3_int_txt(prefix + "_final_GstarAO_gpu.txt", gstar, p.nB, p.nA, p.nO);
        write_matrix_txt(prefix + "_final_BKP_gpu.txt", bkp, p.nB, p.nS);
        write_vector_txt(prefix + "_final_Vnew_gpu.txt", vnew);
    }
};

}  // namespace

// Entry point

int main(int argc, char** argv)
{
    try {
        const BenchOptions opt = parse_bench_args(argc, argv);
        Problem p = load_problem(opt.input_path);
        const std::vector<double> vold0 = compute_initial_vold(p);

        GpuContextV4 ctx(p);
        BenchSummary summary;
        summary.version = "v4_bench_transition_correct";
        summary.input_path = opt.input_path;
        summary.nS = p.nS;
        summary.nA = p.nA;
        summary.nO = p.nO;
        summary.nB = p.nB;
        summary.nG0 = static_cast<int>(p.Gamma.size() / static_cast<size_t>(p.nS));
        summary.iters = opt.iters;
        summary.warmup = opt.warmup;
        summary.runs = opt.runs;

        for (int w = 0; w < opt.warmup; ++w) {
            ctx.reset(p.Gamma, vold0);
            for (int it = 0; it < opt.iters; ++it) ctx.iterate();
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        for (int r = 0; r < opt.runs; ++r) {
            ctx.reset(p.Gamma, vold0);
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaEventRecord(start));
            for (int it = 0; it < opt.iters; ++it) ctx.iterate();
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            const double residual = ctx.fetch_residual();
            summary.run_ms.push_back(static_cast<double>(ms));
            summary.final_residual = residual;
            std::cout << "RUN," << r << ",total_ms=" << ms << ",ms_per_iter=" << (ms / opt.iters)
                      << ",final_residual=" << residual << std::endl;
        }

        finalize_summary(summary);
        write_bench_summary(summary, opt.output_prefix);
        if (opt.dump_final) ctx.dump_final(opt.output_prefix);
        std::cout << "BENCH_SUMMARY,version=" << summary.version
                  << ",mean_total_ms=" << summary.mean_ms
                  << ",mean_ms_per_iter=" << summary.mean_ms_per_iter
                  << ",cov_pct=" << summary.cov_pct
                  << ",final_residual=" << summary.final_residual << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << std::endl;
        return 1;
    }
}

