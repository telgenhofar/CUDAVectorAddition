#include "kernel_benchmark.h"
#include "vecadd.h"
#include "cuda_check.cuh"

double vecAddKernelMs(int n, int repeats) {
    size_t size = n * sizeof(float);
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    vecAddDevice(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) {
        vecAddDevice(A_d, B_d, C_d, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return ms / repeats;
}