# Making Matrix Multiplication Fast in C++
### A step-by-step walkthrough: cache-friendly loops, blocking, SIMD, and packing

> **Who this is for:** anyone curious about how to make code run 50-80x faster *without changing the math*, even if you have never studied computer architecture. Every term is explained when it first appears, and there is a glossary at the end.
>
> **Status:** Steps 1-3 are done and measured. Step 4 (packing) has been written but **not yet measured**. Step 5 (multi-threading) comes next.

---

## Table of contents

1. [The goal](#1-the-goal)
2. [The machine we are running on](#2-the-machine-we-are-running-on)
3. [Background concepts you need](#3-background-concepts-you-need)
4. [Step 1: Baseline and loop reordering](#4-step-1-baseline-and-loop-reordering)
5. [Step 2: Cache blocking (tiling)](#5-step-2-cache-blocking-tiling)
6. [Step 3: SIMD micro-kernel](#6-step-3-simd-micro-kernel)
7. [Step 4: Packing](#7-step-4-packing-written-not-yet-measured)
8. [Summary of all results](#8-summary-of-all-results)
9. [What is next](#9-what-is-next-step-5-multi-threading)
10. [How to build and run](#10-how-to-build-and-run)
11. [Glossary](#11-glossary)

---

## 1. The goal

We want to multiply two big square matrices, **C = A × B**, as fast as possible on one CPU.

A matrix is just a grid of numbers. For N×N matrices, entry `C[i][j]` is computed like this:

```
C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + ... + A[i][N-1]*B[N-1][j]
```

That is "row `i` of A, paired up with column `j` of B, multiply each pair, add everything up". There are N×N entries in C, and each needs N multiplications and N additions. So the total work is about **2 × N³ arithmetic operations**. For N = 1024 that is about 2.1 billion operations.

The textbook algorithm is three nested loops. Everything in this document is about doing **exactly the same arithmetic** but arranging it so the CPU can execute it much more efficiently. The answer does not change (apart from tiny rounding differences, explained later); only the speed does.

**Why this matters beyond matrices:** matrix multiplication sits underneath machine learning, graphics, physics simulation, and scientific computing. The techniques here (cache blocking, SIMD, packing) apply to almost any performance-critical code.

---

## 2. The machine we are running on

From the screenshot you shared:

| Property | Value | What it means |
|---|---|---|
| CPU | **Intel Core Ultra 5 125H** (Meteor Lake) | A modern laptop chip |
| Layout | 4 P-cores + 8 E-cores + 2 low-power E-cores = 14 cores, 18 threads | **Hybrid design**: P-cores ("performance") are big and fast, E-cores ("efficiency") are small and slower |
| SIMD | AVX2 + FMA (no AVX-512) | Can do math on 8 floats at once per instruction |
| L1 cache | 1.38 MiB *total* (about 48 KB per P-core data cache) | Tiny, extremely fast memory next to each core |
| L2 cache | 14 MiB *total* (about 2 MB per P-core) | Bigger, a bit slower |
| L3 cache | 18 MiB, shared by all cores | Bigger still, slower still |
| Governor | `powersave` | The OS lets the clock stay low until the CPU is busy; ramps up under load |

> **Note:** the cache sizes shown are sums over all cores. The per-core figures (48 KB L1 data, ~2 MB L2 on a P-core) are from Intel's published specs for this generation, not from the screenshot.

### Theoretical peak speed (a rough estimate)

A P-core can start **2 FMA instructions per cycle** (FMA = "fused multiply-add", explained below). Each handles 8 floats, and each float-FMA is 2 operations (one multiply, one add):

```
2 instructions/cycle × 8 floats × 2 ops = 32 operations per cycle
32 ops/cycle × ~4.5 GHz (assumed max turbo) ≈ 140 GFLOPS per P-core
```

So **~140 GFLOPS is the ceiling for one P-core**. It is an estimate: the real clock varies with heat and power, and depends on which core the OS puts our program on. Every result below should be judged against this number.

---

## 3. Background concepts you need

If you already know this material, skip ahead. If not, this section is the foundation for everything else.

### 3.1 FLOPS and GFLOPS

**FLOP** = one floating-point operation (one add or one multiply on a decimal number). **GFLOPS** = billions of FLOPs per second.

```
GFLOPS = (2 × N³) / (seconds taken) / 1,000,000,000
```

Why not just report seconds? Because the time depends heavily on N (doubling N makes the work 8× larger). GFLOPS lets you compare different sizes, compare against the hardware's theoretical peak, and diagnose problems:

| GFLOPS level (on this machine) | Typical meaning |
|---|---|
| ~1 | Memory access pattern is terrible; CPU is mostly waiting |
| ~30 | Memory is OK, but the CPU's math units are starved by too many loads/stores |
| ~80 | Good use of SIMD, one core |
| ~140 | Peak of one P-core |

### 3.2 Memory is slow; caches make it bearable

A CPU can do arithmetic far faster than it can fetch numbers from main memory (RAM). To hide this, CPUs have small, fast memories called **caches**, arranged in levels:

```
        speed              size (per P-core, approx.)
        fastest   Registers      ~16 vector registers × 32 bytes = 512 bytes
           |      L1 cache       48 KB      (about 4-5 cycles to access)
           |      L2 cache       ~2 MB      (about 14 cycles)
           |      L3 cache       18 MB, shared (about 50+ cycles)
        slowest   Main memory    GBs       (about 200+ cycles)
```

**The core rule of performance programming: reuse data while it is still in a fast level.** Fetching a number from RAM costs roughly as much as dozens of arithmetic operations, so you want each fetched number to be used many times before it is evicted.

### 3.3 Cache lines

Memory is not fetched one number at a time. It is fetched in **cache lines of 64 bytes** (16 `float`s, since a float is 4 bytes). If you read `x[0]`, you get `x[0]` through `x[15]` for free. So **reading neighbouring numbers is cheap; jumping around memory is expensive**, because every jump pulls in a full 64-byte line while using only 4 bytes of it.

### 3.4 How a 2D matrix is stored in memory (row-major)

Memory is one long line of bytes. A 2D matrix has to be flattened. C++ (and our code) uses **row-major** order: row 0 first, then row 1, and so on.

```
Matrix (N=4):               In memory (one flat array):
  a00 a01 a02 a03           [a00 a01 a02 a03 | a10 a11 a12 a13 | a20 ... ]
  a10 a11 a12 a13            row 0              row 1              row 2
  a20 a21 a22 a23
  a30 a31 a32 a33           Element (i, j) lives at index  i*N + j
```

Consequence: **walking along a row is contiguous (fast); walking down a column jumps N floats each step (slow).** For N=1024, one step down a column is a jump of 4096 bytes, so every element lands in a different cache line.

### 3.5 Registers

A **register** is storage inside the CPU core itself, the fastest memory there is. Arithmetic happens on registers. Data must be loaded from memory into registers first, and results stored back afterwards. Loads and stores take up limited hardware resources ("ports"), so **fewer loads/stores per useful calculation = faster**.

### 3.6 SIMD (Single Instruction, Multiple Data)

Normal ("scalar") arithmetic handles one number per instruction. **SIMD** instructions handle several at once. With **AVX2**, a register is **256 bits wide = 8 floats**, and one instruction operates on all 8 lanes simultaneously:

```
 ymm0 = [ 1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0 ]
 ymm1 = [ 10   10   10   10   10   10   10   10  ]
 vmulps ymm2, ymm0, ymm1    ->   ymm2 = [10 20 30 40 50 60 70 80]   (8 multiplications, 1 instruction)
```

AVX2 provides 16 such registers, named `ymm0` ... `ymm15`.

### 3.7 FMA (fused multiply-add)

`FMA(a, b, c) = a*b + c` as **one instruction**, with one rounding step instead of two. It is the workhorse of matrix multiplication, since the inner step of the algorithm is exactly `sum += a*b`. With AVX2, one FMA instruction does this for 8 floats = 16 operations.

### 3.8 Latency vs throughput

Two different "speeds" of an instruction:
- **Latency:** how many cycles until its *result* is ready (FMA: about 4 cycles).
- **Throughput:** how many such instructions can be *started* per cycle (FMA on a P-core: 2 per cycle).

If each FMA needs the result of the previous one (a dependency chain), you pay the full latency each time: 1 FMA per 4 cycles. To reach 2 per cycle you need about **4 cycles × 2 per cycle = 8 independent FMAs in flight at once**. This is why we will use many independent accumulators in Step 3.

### 3.9 Compiler flags we use

```
g++ -O3 -march=native -std=c++17 matmul.cpp -o matmul
```
- `-O3`: maximum optimization.
- `-march=native`: generate instructions for *this exact CPU*, including AVX2 and FMA. Without it, the compiler would assume a very old CPU (SSE2 only).
- `-std=c++17`: language version (we use `std::aligned_alloc`).

---

## 4. Step 1: Baseline and loop reordering

### 4.1 The naive version (`ijk`)

```cpp
void matmul_ijk(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}
```

This is a direct translation of the math. Look at the innermost loop (over `k`):
- `A[i*N + k]` moves along a **row** of A: contiguous, good.
- `B[k*N + j]` moves down a **column** of B: each step jumps `N` floats (4096 bytes at N=1024). Every access hits a different cache line, and uses only 4 of its 64 bytes. **Bad.**

### 4.2 The reordered version (`ikj`)

We swap the two inner loops. The math is unchanged because addition order within the sum does not matter (up to rounding):

```cpp
void matmul_ikj(const float* A, const float* B, float* C, int N) {
    std::fill(C, C + (size_t)N * N, 0.f);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            float a = A[i * N + k];            // one number, reused across the whole j loop
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
}
```

Now the innermost loop (over `j`) walks along row `i` of C and row `k` of B, both **contiguous**. As a bonus, the compiler can now automatically use SIMD on this loop, because consecutive `j` iterations touch consecutive memory and are independent of each other.

### 4.3 The benchmark harness

```cpp
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
```

It runs the function several times and keeps the **fastest** run. Reasoning: the fastest run is the one least disturbed by other programs, clock ramp-up, or cache warm-up. `steady_clock` is a timer that never goes backwards. We also compute `max_diff`, the largest absolute difference between our result and a reference, to check that we haven't broken correctness.

### 4.4 Your measured output (N = 1024)

```
ijk :    2.002 s     1.07 GFLOPS
ikj :    0.069 s    30.95 GFLOPS  (max diff 9.16e-05)
```

**Interpretation:**
- Reordering the loops, **with zero change to the math**, gave a **29× speedup** (2.002 s → 0.069 s).
- `max diff 9.16e-05` is **not a bug**. Floating-point addition is not perfectly associative (`(a+b)+c` can differ from `a+(b+c)` in the last digits). Changing the order of loops changes the order of additions, so tiny differences (here about 0.00009 on values around 250) are expected.
- At 30.95 GFLOPS we are at about 22% of one core's ~140 GFLOPS peak. The loop is *memory-friendly* now but still not *compute-efficient*; Steps 2-3 address that.

> **Lesson:** the *order* in which you touch memory can matter more than the arithmetic itself.

---

## 5. Step 2: Cache blocking (tiling)

### 5.1 The problem that remains

In `ikj`, for each of the N rows of A, we sweep over **all of B** (N×N floats). At N=2048, B is 2048 × 2048 × 4 bytes = **16 MB**, bigger than L2, and not much smaller than L3. So for every row of A, B has to be streamed in from a slower level again. The same data is fetched N times.

### 5.2 The idea

Chop the matrices into small square **tiles** of size BS×BS. Do all the work that involves one tile of each matrix *while those tiles are sitting in cache*, then move on. A tile gets reused BS times before being evicted instead of once.

```
        B                          Work on one (ii, kk, jj) triple at a time:
   +----+----+----+                 C-tile[ii,jj] += A-tile[ii,kk] × B-tile[kk,jj]
   |    |    |    |                 All three tiles (3 × BS² floats) fit in cache.
   +----+----+----+
   |    |////|    |  <- B-tile[kk,jj]
   +----+----+----+
```

### 5.3 The code

```cpp
void matmul_blocked(const float* __restrict A, const float* __restrict B,
                    float* __restrict C, int N, int BS) {
    std::fill(C, C + (size_t)N * N, 0.f);
    for (int ii = 0; ii < N; ii += BS)          // three outer loops step tile by tile
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS) {
                int iMax = std::min(ii + BS, N);
                int kMax = std::min(kk + BS, N);
                int jMax = std::min(jj + BS, N);
                for (int i = ii; i < iMax; ++i)         // three inner loops: the same
                    for (int k = kk; k < kMax; ++k) {   // ikj code, but confined to one tile
                        float a = A[i * N + k];
                        for (int j = jj; j < jMax; ++j)
                            C[i * N + j] += a * B[k * N + j];
                    }
            }
}
```

`__restrict` is a promise to the compiler that `A`, `B`, `C` do not overlap in memory. Without that promise the compiler must assume writing `C` might change `A` or `B`, which blocks some optimizations.

### 5.4 Your measured output

**N = 1024**
```
ijk :             2.017 s     1.06 GFLOPS
ikj :             0.086 s    24.97 GFLOPS
blocked BS=16 :   0.140 s    15.39 GFLOPS
blocked BS=32 :   0.112 s    19.17 GFLOPS
blocked BS=64 :   0.089 s    24.19 GFLOPS
blocked BS=128:   0.082 s    26.21 GFLOPS
blocked BS=256:   0.072 s    29.94 GFLOPS
```

**N = 2048**
```
ijk :             19.645 s    0.87 GFLOPS
ikj :              0.998 s   17.21 GFLOPS
blocked BS=16 :    1.295 s   13.26 GFLOPS
blocked BS=32 :    0.982 s   17.49 GFLOPS
blocked BS=64 :    0.784 s   21.91 GFLOPS
blocked BS=128:    0.629 s   27.32 GFLOPS
blocked BS=256:    0.588 s   29.20 GFLOPS
```

### 5.5 Interpretation

1. **Blocking did not make the fast case faster. It prevented the slowdown.**
   - At N=1024, `ikj` and blocked are about equal (~25-30), because B (4 MB) still fits in cache.
   - At N=2048, `ikj` **collapses to 17 GFLOPS** (B no longer fits), while blocked BS=256 **holds at 29**. That is the value of blocking.
2. **Bigger tiles won (BS=256 best).** My initial guess was BS=64 would be best. I was wrong. With small tiles, the innermost `j` loop is only 16-32 iterations long, so loop overhead and re-loading `C` dominate. BS=256 uses 3 × 256 × 256 × 4 bytes = 768 KB, which fits in the P-core's ~2 MB L2.
3. **Run-to-run noise is real.** `ikj` at N=1024 measured 30.95 GFLOPS in the first run and 24.97 in the second. Likely causes: the `powersave` frequency governor, and the OS possibly scheduling the program on a slower E-core on this hybrid CPU. For stable numbers, pin the program to a P-core (see [Section 10](#10-how-to-build-and-run)).
4. **Everything is stuck around 30 GFLOPS.** The bottleneck is no longer cache misses. It is the *instruction mix*, which Step 3 fixes.

---

## 6. Step 3: SIMD micro-kernel

### 6.1 Why the current inner loop is slow

Even with good cache behavior, the inner loop does this for every 8 floats:

```
load   B[k][j..j+7]       <- 1 load
load   C[i][j..j+7]       <- 1 load
FMA    C = C + a*B        <- 1 useful math instruction
store  C[i][j..j+7]       <- 1 store
```

That is **2 loads + 1 store for each FMA**. The chip's load/store ports are the bottleneck; the 2 FMA units sit mostly idle. It is like a chef who can chop 2 vegetables per second but only gets one delivered every 3 seconds.

### 6.2 The fix: keep a small block of C in registers

The result `C[i][j]` is *accumulated* over all N values of `k`. There is no reason to write it back to memory after every `k`. Instead:

1. Load a small block of C into registers **once**.
2. Loop over `k`, doing only FMAs on those registers. Memory traffic is just the inputs `A` and `B`.
3. Store the block back **once**.

This small, register-resident piece of work is called a **micro-kernel**. It is the heart of every high-performance matrix library (BLAS).

### 6.3 Intrinsics: how to write specific CPU instructions from C++

**Intrinsics** are functions that map almost 1:1 onto machine instructions. A `__m256` variable means "a value living in a 256-bit register" (8 floats).

| Intrinsic | Assembly instruction | What it does |
|---|---|---|
| `_mm256_loadu_ps(p)` | `vmovups ymm, [p]` | Load 8 consecutive floats from memory (`u` = address need not be aligned) |
| `_mm256_broadcast_ss(p)` | `vbroadcastss ymm, [p]` | Load **one** float and copy it into all 8 lanes |
| `_mm256_fmadd_ps(a,b,c)` | `vfmadd231ps` | `a*b + c` on 8 lanes at once |
| `_mm256_storeu_ps(p,v)` | `vmovups [p], ymm` | Store 8 floats to memory |

### 6.4 The 6×16 micro-tile

The micro-kernel computes a **6-row × 16-column** tile of C. That is 6 × 16 = 96 numbers = 12 registers' worth (since each register holds 8 floats: 6 rows × 2 registers per row).

For each step `k`:
```
b0 = load B[k][0..7]            ┐ two loads for B: the same 16 columns of row k
b1 = load B[k][8..15]           ┘
for each of the 6 rows r:
    a  = broadcast A[r][k]       <- one load; the same A value is used for all 16 columns
    c0[r] += a * b0              <- FMA
    c1[r] += a * b1              <- FMA
```

Per `k`: **8 loads** (2 + 6) feed **12 FMAs**. Before: 12 FMAs needed 24 loads/stores. Memory traffic per FMA fell by 3×.

```
          B row k (16 floats)
          [ b0 (8) | b1 (8) ]
              ×        ×
 A[0][k] -> [ c0[0] | c1[0] ]      Each row r of the 6×16 tile of C
 A[1][k] -> [ c0[1] | c1[1] ]      is held in two registers: c0[r], c1[r].
 A[2][k] -> [ c0[2] | c1[2] ]
 A[3][k] -> [ c0[3] | c1[3] ]
 A[4][k] -> [ c0[4] | c1[4] ]
 A[5][k] -> [ c0[5] | c1[5] ]
```

### 6.5 Why exactly 6×16? Three constraints

1. **Register budget.** 12 accumulators + 2 (`b0`, `b1`) + 1 (broadcast `a`) = **15 of 16** registers. If you needed more than 16, the compiler would spill to the stack (memory) and performance would collapse.
2. **Latency hiding.** Section 3.8: we need ~8 independent FMAs in flight. Each of our 12 accumulators is an independent chain, so there are 12. Enough.
3. **Load bandwidth.** 8 loads per 12 FMAs (= 6 cycles at 2 FMA/cycle) is about 1.3 loads per cycle, which the load ports can sustain.

### 6.6 The code

```cpp
constexpr int MR = 6;    // rows of the micro-tile
constexpr int NR = 16;   // columns of the micro-tile (2 vectors of 8)

template <int M>      // M = number of rows actually handled (6 normally, fewer at the matrix edge)
__attribute__((noinline))
void microkernel(const float* A, const float* B, float* C,
                 int kc, int lda, int ldb, int ldc) {
    __m256 c0[M], c1[M];                      // accumulators; the compiler keeps them in registers

    #pragma GCC unroll 8
    for (int r = 0; r < M; ++r) {             // load the C tile ONCE
        c0[r] = _mm256_loadu_ps(C + r * ldc);
        c1[r] = _mm256_loadu_ps(C + r * ldc + 8);
    }

    for (int k = 0; k < kc; ++k) {            // the hot loop: only loads that feed FMAs
        __m256 b0 = _mm256_loadu_ps(B + k * ldb);
        __m256 b1 = _mm256_loadu_ps(B + k * ldb + 8);
        #pragma GCC unroll 8
        for (int r = 0; r < M; ++r) {
            __m256 a = _mm256_broadcast_ss(A + r * lda + k);
            c0[r] = _mm256_fmadd_ps(a, b0, c0[r]);
            c1[r] = _mm256_fmadd_ps(a, b1, c1[r]);
        }
    }

    #pragma GCC unroll 8
    for (int r = 0; r < M; ++r) {             // store the C tile ONCE
        _mm256_storeu_ps(C + r * ldc,     c0[r]);
        _mm256_storeu_ps(C + r * ldc + 8, c1[r]);
    }
}
```

Detail notes:
- **`template <int M>`**: the row count is a compile-time constant, so the compiler can fully unroll the `r` loops and assign each `c0[r]`, `c1[r]` to a fixed register. With a runtime count it could not.
- **`#pragma GCC unroll 8`**: tells the compiler to copy-paste the loop body 8 times instead of looping. Arrays can only live in registers if every index is a known constant.
- **`noinline`**: keeps this function as its own named symbol so you can find and read its assembly. The call overhead is trivial: one call per `12 × kc` FMAs.
- **`lda`, `ldb`, `ldc`** ("leading dimension"): the row stride of each matrix, i.e. how many floats to skip to get to the next row. Here all equal N.

### 6.7 The outer loops: mapping onto the cache hierarchy

```cpp
void matmul_simd(const float* A, const float* B, float* C, int N, int MC, int KC, int NC) {
    std::fill(C, C + (size_t)N * N, 0.f);
    for (int jc = 0; jc < N; jc += NC)                // slice of columns
        for (int pc = 0; pc < N; pc += KC)            // slice of the k dimension
            for (int ic = 0; ic < N; ic += MC)        // slice of rows
                for (int jr = 0; jr < nc; jr += NR)         // 16-column micro-panel
                    for (int ir = 0; ir < mc; ir += MR)     // 6-row micro-panel
                        microkernel(...);   // (dispatch handles edge rows)
}
```
(Abbreviated. `mc`, `nc`, `kc` are the actual sizes clipped at the matrix edge.)

| Data | Approx. size (MC=192, KC=256) | Lives in | Reused by |
|---|---|---|---|
| C micro-tile (6×16) | 12 registers | **Registers** | all `kc` steps |
| B micro-panel (kc × 16) | 16 KB | **L1** (48 KB) | every `ir` iteration |
| A block (mc × kc) | 192 KB | **L2** (~2 MB) | every `jr` iteration |

Each loop level is chosen so that the data it keeps re-using fits in the matching level of the memory hierarchy. This three-level structure is the standard design of high-performance matrix multiplication.

`dispatch(m, ...)` is a `switch` that calls `microkernel<6>` normally, or `microkernel<5>` ... `<1>` for leftover rows when N is not divisible by 6. In this version N must be a multiple of 16 (column edge cases are omitted for simplicity).

### 6.8 Your measured output

Blocks below are, in order, N = 1008, N = 1024, N = 2048 (I'm inferring this order from the values).

**N = 1008**
```
ikj                   :   0.074 s   29.03 GFLOPS
blocked BS=256        :   0.065 s   32.90 GFLOPS (max diff 0.00e+00)
simd MC=96  KC=128    :   0.026 s   81.25 GFLOPS
simd MC=192 KC=128    :   0.026 s   81.87 GFLOPS
simd MC=96  KC=256    :   0.026 s   82.76 GFLOPS
simd MC=192 KC=256    :   0.026 s   81.29 GFLOPS
simd MC=96  KC=512    :   0.035 s   60.52 GFLOPS
simd MC=192 KC=512    :   0.032 s   67.25 GFLOPS
```
**N = 1024**
```
ikj                   :   0.074 s   27.75 GFLOPS
blocked BS=256        :   0.067 s   30.48 GFLOPS
simd MC=96  KC=128    :   0.025 s   81.26 GFLOPS
simd MC=192 KC=128    :   0.025 s   82.23 GFLOPS
simd MC=96  KC=256    :   0.026 s   77.44 GFLOPS
simd MC=192 KC=256    :   0.025 s   81.23 GFLOPS
simd MC=96  KC=512    :   0.026 s   78.76 GFLOPS
simd MC=192 KC=512    :   0.026 s   79.61 GFLOPS
```
**N = 2048**
```
ikj                   :   0.926 s   18.56 GFLOPS
blocked BS=256        :   0.596 s   28.83 GFLOPS
simd MC=96  KC=128    :   0.243 s   70.65 GFLOPS
simd MC=192 KC=128    :   0.241 s   71.17 GFLOPS
simd MC=96  KC=256    :   0.267 s   64.46 GFLOPS
simd MC=192 KC=256    :   0.262 s   65.55 GFLOPS
simd MC=96  KC=512    :   0.328 s   52.35 GFLOPS
simd MC=192 KC=512    :   0.343 s   50.14 GFLOPS
```
(Every row shows `max diff 0.00e+00` against the `ikj` reference.)

### 6.9 Interpretation

1. **SIMD gave about 2.5-3× over the previous best** (about 30 → about 82 GFLOPS at N ≈ 1000). That's roughly 55-60% of one P-core's ~140 GFLOPS peak. This single change (keeping C in registers) was worth more than all the cache work before it.
2. **Why `max diff` is exactly 0.** `ikj` and the micro-kernel both add up each `C[i][j]` over `k` in the same order (k = 0, 1, 2, ...), and GCC fuses `a*b + c` into FMA by default, so the floating-point operations happen identically. That is why the earlier comparison against `ijk` (different summation order) showed ~9e-5, and these show exactly 0.
3. **N=2048 degrades**: best 71 GFLOPS, and KC=512 falls to 50. Larger N means the rows of B are farther apart in memory (8 KB at N=2048), and the micro-kernel reads the same 16 columns from `kc` different rows. Our *hypotheses* (not yet proven by measurement) are:
   - **Cache-set conflicts:** an L1 cache is divided into "sets"; an address's set is determined by bits of its address, so addresses exactly 4096 bytes apart (or multiples) all compete for the *same set*, which only has 12 slots ("ways") on this CPU. When row spacing is a multiple of 4096, the `kc` lines of the B panel all fight over one set and keep evicting each other.
   - **TLB pressure:** the TLB caches virtual→physical address translations, one per 4 KB page. At 8 KB row spacing, each of the `kc` rows is on a different page, so a 512-row panel needs 512 translations; the first-level TLB holds only about 100.
4. **A wrong prediction of mine, for honesty.** I predicted N=1024 would look noticeably worse than N=1008 because of the set-conflict effect. At KC=128 and 256 they're equal. The conflicts exist but the L2 cache and hardware prefetcher hid most of the cost at that size. The slow KC=512 result at N=1008 (60-67 GFLOPS vs ~79 at N=1024) I cannot explain from this data; treat it as noise until it reproduces.
5. **We have not yet confirmed from the assembly** that the 12 accumulators stayed in registers. The 82 GFLOPS suggests they did. To check: generate assembly (`g++ ... -S -masm=intel`), find the `microkernelILi6E` function, and confirm the inner loop has about 12 `vfmadd231ps` instructions and no `[rsp+...]` memory operands.

> **Lesson:** SIMD alone is not enough. What mattered was *reducing memory instructions per useful FMA* by holding results in registers.

---

## 7. Step 4: Packing (written, NOT yet measured)

### 7.1 The idea

Problem (from Section 6.9): the micro-kernel reads a small piece of B (16 columns × `kc` rows), but in the original matrix those pieces are `N × 4` bytes apart. That scattered layout causes cache-set conflicts and TLB pressure.

**Solution:** before computing, **copy** the blocks of A and B into temporary buffers, laid out in exactly the order the micro-kernel will read them. The kernel then reads one straight, contiguous stream.

```
Original B (row-major, stride N):          Packed B (panel-major):
 row k:    [ ...16 floats... ] ... far ...  Bp = [ k=0: 16 floats ][ k=1: 16 floats ][ k=2: ... ]
 row k+1:  [ ...16 floats... ] ... away     all contiguous: a kc=256 panel is one 16 KB chunk
```

Benefits of the packed layout:
- A 16 KB contiguous panel spreads **evenly across all 64 cache sets**, so no conflicts.
- It spans only ~4 pages instead of 256+, so TLB pressure vanishes.
- Contiguous memory is easy for the hardware prefetcher to predict and fetch ahead of time.
- The kernel's address arithmetic becomes trivial (`Bp + k*16`, `Ap + k*6`).

**Cost:** copying is O(N²) work versus O(N³) for the multiplication, so it becomes negligible as N grows.

### 7.2 Packed layouts

```
A block (mc × kc)  ->  panels of 6 rows, layout [k][r]:   Ap[panel][k][0..5]
   so for each k the kernel reads 6 consecutive floats (one per row)

B block (kc × nc)  ->  panels of 16 columns, layout [k][j]: Bp[panel][k][0..15]
   so for each k the kernel reads 16 consecutive floats (two ymm loads)
```

### 7.3 The code

```cpp
static void pack_A(const float* A, int lda, float* Ap, int mc, int kc) {
    for (int ir = 0; ir < mc; ir += MR) {
        int m = std::min(MR, mc - ir);
        float* dst = Ap + (size_t)(ir / MR) * kc * MR;
        for (int r = 0; r < m; ++r) {
            const float* src = A + (size_t)(ir + r) * lda;   // contiguous read
            for (int k = 0; k < kc; ++k)
                dst[k * MR + r] = src[k];                    // strided write (transpose-like)
        }
    }
}

static void pack_B(const float* B, int ldb, float* Bp, int kc, int nc) {
    for (int jr = 0; jr < nc; jr += NR) {
        float* dst = Bp + (size_t)(jr / NR) * kc * NR;
        for (int k = 0; k < kc; ++k)
            for (int j = 0; j < NR; ++j)
                dst[k * NR + j] = B[(size_t)k * ldb + jr + j];
    }
}
```

The packed micro-kernel is identical to Section 6.6 except for addressing: it reads `Bp + k*NR` and `Ap + k*MR + r`, and no longer needs `lda`/`ldb` parameters.

```cpp
static float* alloc64(size_t floats) {         // 64-byte aligned allocation = starts on a cache-line boundary
    size_t bytes = (floats * sizeof(float) + 63) & ~size_t(63);  // aligned_alloc needs size % alignment == 0
    return static_cast<float*>(std::aligned_alloc(64, bytes));
}
```

Why align to 64 bytes? A cache line is 64 bytes. If a 16-float row of `Bp` started mid-line, it would straddle two lines and need two fetches. Aligned, it is exactly one line.

The driver (`matmul_packed`) uses the same five-level loop nest as before, but `pack_B` runs once per `(jc, pc)` pair and `pack_A` once per `(jc, pc, ic)` triple, writing into the buffers `Bp` (about 512 KB at KC=256, NC=512) and `Ap` (about 192 KB at MC=192, KC=256), both of which fit in L2.

### 7.4 What I expect (predictions, to be verified)

- **N=1008/1024:** modest gain, maybe 82 → 90-100 GFLOPS.
- **N=2048:** the main win; I expect it to recover to the small-N level instead of 50-70, and the KC=512 penalty should shrink or vanish.
- If N=2048 stays well below N=1024, then my set-conflict/TLB explanation is incomplete and we need to investigate (for instance with `perf stat` counters) before adding threads.

---

## 8. Summary of all results

Best result of each technique, in GFLOPS (higher is better), single thread:

| Technique | N=1024 | N=2048 | Key idea |
|---|---|---|---|
| Naive `ijk` | ~1.07 | ~0.87 | Direct translation of the formula; column walks wreck the cache |
| `ikj` loop order | 25-31 | 17-19 | Contiguous inner loop; compiler auto-vectorizes |
| Cache blocking (BS=256) | 30-33 | 29 | Tiles stay in cache; no collapse at large N |
| SIMD micro-kernel (best config) | ~82 | ~71 | C block in registers; 12 FMAs per 8 loads |
| + Packing | *not yet measured* | *not yet measured* | Contiguous panels; no conflicts |
| One P-core theoretical peak | ~140 | ~140 | 2 FMA/cycle × 8 lanes × 2 ops × ~4.5 GHz (estimate) |

Cumulative speedup so far at N=1024: 1.07 → ~82 GFLOPS is about **75× faster** than the naive code, with identical results.

```
GFLOPS (N=1024)
  naive     |▏                                 1
  ikj       |████                              ~30
  blocked   |████                              ~30
  simd      |████████████▌                     ~82
  peak(est) |█████████████████████▍            ~140
```

**What each step really taught:**
1. **Memory access order** (Step 1): contiguous access is worth ~29×.
2. **Cache reuse** (Step 2): keeps performance from degrading as data grows.
3. **Instruction mix** (Step 3): fewer loads/stores per FMA lets the math units finally work (~2.7×).
4. **Data layout** (Step 4): arranging data to match how it will be read removes hardware-level conflicts (pending measurement).

---

## 9. What is next: Step 5, multi-threading

So far everything runs on **one core**. This chip has 4 P-cores (8 threads) plus 10 E-cores. We will use **OpenMP** (a compiler feature: `#pragma omp parallel for`) to split the work.

The packing structure tells us how to split:
- Parallelize the `ic` loop (row blocks of A). Different threads write to different rows of C, so they never conflict.
- Each thread gets its **own** `Ap` buffer; all threads **share** one `Bp` (read-only).
- Because the cores are not equal (P vs E), we will use **dynamic scheduling**: threads grab the next chunk of work when they finish, so fast cores simply do more chunks instead of waiting on slow ones.

Expected result: several hundred GFLOPS, bounded by the sum of all cores and by memory bandwidth and shared cache.

---

## 10. How to build and run

Compile:
```bash
g++ -O3 -march=native -std=c++17 matmul.cpp -o matmul
```

Run (pinned to one core so the OS does not move it between P- and E-cores):
```bash
lscpu -e                      # look for the CPUs with the highest MAXMHZ; those are P-cores
taskset -c 0 ./matmul 1008    # replace 0 with a P-core number; N must be a multiple of 16
taskset -c 0 ./matmul 1024
taskset -c 0 ./matmul 2048
```

Optional, for steadier timings:
```bash
sudo cpupower frequency-set -g performance    # stop the CPU from idling its clock down
```

Inspect the generated assembly:
```bash
g++ -O3 -march=native -std=c++17 -S -masm=intel matmul.cpp -o matmul.s
# open matmul.s, search for "microkernelILi6E", look at the inner loop:
#   want:    ~12 x vfmadd231ps, with vbroadcastss / vmovups feeding them
#   avoid:   any [rsp+...] operand inside the loop (means registers spilled to the stack)
```

---

## 11. Glossary

| Term | Plain-English meaning |
|---|---|
| **AVX2** | CPU instruction set with 256-bit vector registers (8 floats at once) |
| **Associativity ("ways")** | How many cache lines can share one cache "set" before one must be evicted |
| **Broadcast** | Copy one value into every lane of a vector register |
| **Cache** | Small, fast memory between the CPU and RAM that holds recently used data |
| **Cache line** | The 64-byte unit in which memory is moved into cache |
| **Cache blocking / tiling** | Restructuring loops to work on small pieces that fit in cache |
| **Conflict miss** | A cache miss caused by too many addresses mapping to the same set, even though the cache has free space elsewhere |
| **Core (P / E)** | An independent processor inside the chip. P = fast "performance" core; E = smaller "efficiency" core |
| **Dependency chain** | A sequence of operations where each needs the previous result, so they cannot overlap |
| **Dynamic scheduling** | Handing out work pieces on demand, so faster workers do more |
| **FMA** | Fused multiply-add: `a*b + c` in one instruction |
| **FLOP / GFLOPS** | Floating-point operation / billions of them per second |
| **Hardware prefetcher** | CPU logic that detects sequential access and loads data before it is requested |
| **Intrinsic** | A C++ function that compiles to one specific CPU instruction |
| **Latency** | Cycles until an instruction's result is available |
| **Leading dimension (`ld*`)** | Number of floats between the starts of consecutive rows of a matrix |
| **Micro-kernel** | The innermost, register-resident computation (here 6×16 of C) |
| **OpenMP** | A compiler feature for easy multi-threading via `#pragma` directives |
| **Packing** | Copying data into a contiguous buffer in the order it will be consumed |
| **Page / TLB** | Memory is divided into 4 KB pages; the TLB is a small cache of page address translations |
| **Register** | Storage inside the CPU core; the fastest memory; arithmetic operates on registers |
| **Row-major** | Storing a matrix row by row in memory |
| **SIMD** | Single Instruction, Multiple Data: one instruction operates on a whole vector |
| **Spill** | When the compiler runs out of registers and stores variables on the stack instead |
| **Throughput** | How many instructions can be started per cycle |
| **`__restrict`** | A promise to the compiler that pointers do not overlap |