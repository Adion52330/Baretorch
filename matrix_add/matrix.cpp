#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <omp.h>

using namespace std;

// __mm256 is 256 bit register which can store 8 floats. when we add two floats, we can add 8 floats at a time using AVX2 intrinsics. This is called SIMD (Single Instruction Multiple Data) parallelism.

void matrix_add(
    const float* A,
    const float* B,
    float* C,
    size_t size
) {
    // size_t i = 0;
    // // Process 8 floats at a time
    // for (; i + 8 <= size; i += 8) {

    //     __m256 a = _mm256_loadu_ps(A + i);

    //     __m256 b = _mm256_loadu_ps(B + i);

    //     __m256 c = _mm256_add_ps(a, b);

    //     _mm256_storeu_ps(C + i, c);
    // }

    // for (; i < size; i++) {
    //     C[i] = A[i] + B[i];
    // }

    /*
        using multiple cpu cores to add two matrices. This is called multi-threading. We can use OpenMP to parallelize the for loop. OpenMP is a library that allows us to write parallel code in C++. We can use the #pragma omp parallel for directive to tell the compiler to parallelize the for loop. The compiler will then generate code that will run the for loop in parallel on multiple threads. Each thread will execute a portion of the for loop. The number of threads is determined by the number of CPU cores available on the machine. The default number of threads is equal to the number of CPU cores. We can change the number of threads by setting the OMP_NUM_THREADS environment variable.
    */
    // #pragma omp parallel for
    // for (size_t i = 0; i < size; i++) {
    //     C[i] = A[i] + B[i];
    // }

    #pragma omp parallel for

    for (size_t i = 0; i < size; i += 8) {

        __m256 a = _mm256_loadu_ps(A + i);

        __m256 b = _mm256_loadu_ps(B + i);

        __m256 c = _mm256_add_ps(a, b);

        _mm256_storeu_ps(C + i, c);
    }

}

int main() {
    const size_t N = 4096;

    vector<float> A(N * N);
    vector<float> B(N * N);
    vector<float> C(N * N);

    // Initialize
    for (size_t i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    auto start = chrono::high_resolution_clock::now();

    matrix_add(A.data(), B.data(), C.data(), N*N);

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    cout << "Time: " << elapsed.count() << " seconds\n";
    cout << "C[0] = " << C[0] << '\n';
}