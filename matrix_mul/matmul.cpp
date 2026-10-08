// matmul.cpp
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <immintrin.h>
#include <algorithm>
#include <omp.h>

using Mat = std::vector<float>;

// Naive: i-j-k. Inner loop walks down a column of B (stride N) -> cache-hostile.
void matmul_ijk(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// Reordered: i-k-j. Inner loop walks rows of B and C contiguously.
void matmul_ikj(const float* A, const float* B, float* C, int N) {
    std::fill(C, C + (size_t)N * N, 0.f);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            float a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
}

template <typename F>
double bench(F f, const Mat& A, const Mat& B, Mat& C, int N, int reps = 3) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        f(A.data(), B.data(), C.data(), N);
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

float max_diff(const Mat& X, const Mat& Y) {
    float d = 0;
    for (size_t i = 0; i < X.size(); ++i) d = std::max(d, std::fabs(X[i] - Y[i]));
    return d;
}

// Cache-blocked i-k-j. Tiles of A, B, C (BS x BS floats each) stay hot in cache.
void matmul_blocked(const float* __restrict A, const float* __restrict B,
                    float* __restrict C, int N, int BS) {
    std::fill(C, C + (size_t)N * N, 0.f);
    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS) {
                int iMax = std::min(ii + BS, N);
                int kMax = std::min(kk + BS, N);
                int jMax = std::min(jj + BS, N);
                for (int i = ii; i < iMax; ++i)
                    for (int k = kk; k < kMax; ++k) {
                        float a = A[i * N + k];
                        for (int j = jj; j < jMax; ++j)
                            C[i * N + j] += a * B[k * N + j];
                    }
            }
}

constexpr int MR = 6;    // rows of C per micro-tile
constexpr int NR = 16;   // cols of C per micro-tile (2 x __m256)

// C[0:M, 0:16] += A[0:M, 0:kc] * B[0:kc, 0:16]
// lda/ldb/ldc = row strides (in floats) of the full matrices.
// noinline: gives us a clean symbol to inspect in the assembly. The call cost
// is negligible: one call per 12*kc FMAs.
template <int M>
__attribute__((noinline))
void microkernel(const float* A, const float* B, float* C,
                 int kc, int lda, int ldb, int ldc) {
    __m256 c0[M], c1[M];                       // accumulators; should live in registers

    #pragma GCC unroll 8
    for (int r = 0; r < M; ++r) {              // load the C tile once
        c0[r] = _mm256_loadu_ps(C + r * ldc);
        c1[r] = _mm256_loadu_ps(C + r * ldc + 8);
    }

    for (int k = 0; k < kc; ++k) {
        __m256 b0 = _mm256_loadu_ps(B + k * ldb);        // B[k][0..7]
        __m256 b1 = _mm256_loadu_ps(B + k * ldb + 8);    // B[k][8..15]
        #pragma GCC unroll 8
        for (int r = 0; r < M; ++r) {
            __m256 a = _mm256_broadcast_ss(A + r * lda + k);   // A[r][k] -> 8 lanes
            c0[r] = _mm256_fmadd_ps(a, b0, c0[r]);
            c1[r] = _mm256_fmadd_ps(a, b1, c1[r]);
        }
    }

    #pragma GCC unroll 8
    for (int r = 0; r < M; ++r) {              // store the C tile once
        _mm256_storeu_ps(C + r * ldc,     c0[r]);
        _mm256_storeu_ps(C + r * ldc + 8, c1[r]);
    }
}

// Handles the leftover rows at the bottom (N isn't a multiple of 6).
static void dispatch(int m, const float* a, const float* b, float* c,
                     int kc, int lda, int ldb, int ldc) {
    switch (m) {
        case 6: microkernel<6>(a, b, c, kc, lda, ldb, ldc); break;
        case 5: microkernel<5>(a, b, c, kc, lda, ldb, ldc); break;
        case 4: microkernel<4>(a, b, c, kc, lda, ldb, ldc); break;
        case 3: microkernel<3>(a, b, c, kc, lda, ldb, ldc); break;
        case 2: microkernel<2>(a, b, c, kc, lda, ldb, ldc); break;
        case 1: microkernel<1>(a, b, c, kc, lda, ldb, ldc); break;
    }
}

// Requires N % 16 == 0 (column edge handling omitted for clarity).
void matmul_simd(const float* A, const float* B, float* C, int N,
                 int MC, int KC, int NC) {
    std::fill(C, C + (size_t)N * N, 0.f);
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < N; pc += KC) {
            int kc = std::min(KC, N - pc);
            for (int ic = 0; ic < N; ic += MC) {
                int mc = std::min(MC, N - ic);
                for (int jr = 0; jr < nc; jr += NR)           // B micro-panel: stays in L1
                    for (int ir = 0; ir < mc; ir += MR) {     // A block: stays in L2
                        const float* a = A + (size_t)(ic + ir) * N + pc;
                        const float* b = B + (size_t)pc * N + jc + jr;
                        float*       c = C + (size_t)(ic + ir) * N + jc + jr;
                        dispatch(std::min(MR, mc - ir), a, b, c, kc, N, N, N);
                    }
            }
        }
    }
}

// A micro-panel is 6 rows x kc: layout [k][r], so the kernel reads 6 consecutive floats per k.
// Pack rows [0,mc) x cols [0,kc) of A (row stride lda) into panels of MR rows.
static void pack_A(const float* A, int lda, float* Ap, int mc, int kc) {
    for (int ir = 0; ir < mc; ir += MR) {
        int m = std::min(MR, mc - ir);
        float* dst = Ap + (size_t)(ir / MR) * kc * MR;   // fixed stride per panel
        for (int r = 0; r < m; ++r) {
            const float* src = A + (size_t)(ir + r) * lda;   // contiguous read
            for (int k = 0; k < kc; ++k)
                dst[k * MR + r] = src[k];                    // strided write (stride 6 floats)
        }
    }
}

// A B micro-panel is kc x 16: layout [k][j], 16 consecutive floats (2 ymm) per k.
// Requires nc % 16 == 0.
static void pack_B(const float* B, int ldb, float* Bp, int kc, int nc) {
    for (int jr = 0; jr < nc; jr += NR) {
        float* dst = Bp + (size_t)(jr / NR) * kc * NR;
        for (int k = 0; k < kc; ++k)
            for (int j = 0; j < NR; ++j)
                dst[k * NR + j] = B[(size_t)k * ldb + jr + j];   // compiler vectorizes this
    }
}

template <int M>
__attribute__((noinline))
void microkernel_packed(const float* Ap, const float* Bp, float* C, int kc, int ldc) {
    __m256 c0[M], c1[M];
    #pragma GCC unroll 8
    for (int r = 0; r < M; ++r) {
        c0[r] = _mm256_loadu_ps(C + r * ldc);
        c1[r] = _mm256_loadu_ps(C + r * ldc + 8);
    }
    for (int k = 0; k < kc; ++k) {
        __m256 b0 = _mm256_loadu_ps(Bp + k * NR);          // contiguous
        __m256 b1 = _mm256_loadu_ps(Bp + k * NR + 8);
        #pragma GCC unroll 8
        for (int r = 0; r < M; ++r) {
            __m256 a = _mm256_broadcast_ss(Ap + k * MR + r);   // contiguous across k
            c0[r] = _mm256_fmadd_ps(a, b0, c0[r]);
            c1[r] = _mm256_fmadd_ps(a, b1, c1[r]);
        }
    }
    #pragma GCC unroll 8
    for (int r = 0; r < M; ++r) {
        _mm256_storeu_ps(C + r * ldc,     c0[r]);
        _mm256_storeu_ps(C + r * ldc + 8, c1[r]);
    }
}

static void dispatch_packed(int m, const float* ap, const float* bp, float* c, int kc, int ldc) {
    switch (m) {
        case 6: microkernel_packed<6>(ap, bp, c, kc, ldc); break;
        case 5: microkernel_packed<5>(ap, bp, c, kc, ldc); break;
        case 4: microkernel_packed<4>(ap, bp, c, kc, ldc); break;
        case 3: microkernel_packed<3>(ap, bp, c, kc, ldc); break;
        case 2: microkernel_packed<2>(ap, bp, c, kc, ldc); break;
        case 1: microkernel_packed<1>(ap, bp, c, kc, ldc); break;
    }
}

static float* alloc64(size_t floats) {            // 64-byte aligned = cache-line aligned
    size_t bytes = (floats * sizeof(float) + 63) & ~size_t(63);   // size must be a multiple of alignment
    return static_cast<float*>(std::aligned_alloc(64, bytes));
}

void matmul_packed(const float* A, const float* B, float* C, int N,
                   int MC, int KC, int NC) {
    std::fill(C, C + (size_t)N * N, 0.f);
    float* Ap = alloc64((size_t)((MC + MR - 1) / MR) * MR * KC);
    float* Bp = alloc64((size_t)KC * NC);

    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < N; pc += KC) {
            int kc = std::min(KC, N - pc);
            pack_B(B + (size_t)pc * N + jc, N, Bp, kc, nc);       // once per (jc,pc)
            for (int ic = 0; ic < N; ic += MC) {
                int mc = std::min(MC, N - ic);
                pack_A(A + (size_t)ic * N + pc, N, Ap, mc, kc);   // once per (jc,pc,ic)
                for (int jr = 0; jr < nc; jr += NR)               // B panel (16 KB): L1
                    for (int ir = 0; ir < mc; ir += MR)           // A panels stream from L2
                        dispatch_packed(std::min(MR, mc - ir),
                                        Ap + (size_t)(ir / MR) * kc * MR,
                                        Bp + (size_t)(jr / NR) * kc * NR,
                                        C + (size_t)(ic + ir) * N + jc + jr, kc, N);
            }
        }
    }
    std::free(Ap);
    std::free(Bp);
}

static void pack_B_panel(const float* B, int ldb, float* dst, int kc, int jr) {
    for (int k = 0; k < kc; ++k)
        for (int j = 0; j < NR; ++j)
            dst[k * NR + j] = B[(size_t)k * ldb + jr + j];
}

void matmul_parallel(const float* A, const float* B, float* C, int N,
                     int MC, int KC, int NC) {
    std::fill(C, C + (size_t)N * N, 0.f);
    float* Bp = alloc64((size_t)KC * NC);                    // shared
    #pragma omp parallel
    {
        float* Ap = alloc64((size_t)((MC + MR - 1) / MR) * MR * KC);   // private
        for (int jc = 0; jc < N; jc += NC) {
            int nc = std::min(NC, N - jc);
            for (int pc = 0; pc < N; pc += KC) {
                int kc = std::min(KC, N - pc);
                #pragma omp for schedule(static)             // pack B in parallel
                for (int jr = 0; jr < nc; jr += NR)
                    pack_B_panel(B + (size_t)pc * N + jc, N,
                                 Bp + (size_t)(jr / NR) * kc * NR, kc, jr);
                // implicit barrier: Bp complete before anyone computes
                #pragma omp for schedule(dynamic, 1)         // row blocks handed out on demand
                for (int ic = 0; ic < N; ic += MC) {
                    int mc = std::min(MC, N - ic);
                    pack_A(A + (size_t)ic * N + pc, N, Ap, mc, kc);
                    for (int jr = 0; jr < nc; jr += NR)
                        for (int ir = 0; ir < mc; ir += MR)
                            dispatch_packed(std::min(MR, mc - ir),
                                            Ap + (size_t)(ir / MR) * kc * MR,
                                            Bp + (size_t)(jr / NR) * kc * NR,
                                            C + (size_t)(ic + ir) * N + jc + jr, kc, N);
                }
                // implicit barrier: nobody still reading Bp when it gets overwritten
            }
        }
        std::free(Ap);
    }
    std::free(Bp);
}

int main(int argc, char** argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : 1024;
    if (N % 16 != 0) { std::fprintf(stderr, "N must be a multiple of 16\n"); return 1; }
    Mat A(N * N), B(N * N), C(N * N), Ref(N * N);
    for (auto& x : A) x = rand() / (float)RAND_MAX;
    for (auto& x : B) x = rand() / (float)RAND_MAX;
    double flops = 2.0 * N * N * N;

    double t = bench(matmul_ikj, A, B, Ref, N);
    std::printf("%-26s: %7.3f s %8.2f GFLOPS\n", "ikj (reference)", t, flops / t / 1e9);

    auto run = [&](const char* name, auto f) {
        double tt = bench(f, A, B, C, N, 5);
        std::printf("%-26s: %7.3f s %8.2f GFLOPS (max diff %.2e)\n",
                    name, tt, flops / tt / 1e9, max_diff(C, Ref));
    };

    run("packed, 1 thread 96/512", [](const float* a, const float* b, float* c, int n) {
        matmul_packed(a, b, c, n, 96, 512, 512);
    });

    for (int MC : {48, 96})
        for (int T : {1, 2, 4, 8, 14, 18}) {
            omp_set_num_threads(T);
            char name[64];
            std::snprintf(name, sizeof name, "parallel MC=%d threads=%d", MC, T);
            run(name, [=](const float* a, const float* b, float* c, int n) {
                matmul_parallel(a, b, c, n, MC, 512, 512);
            });
        }
}