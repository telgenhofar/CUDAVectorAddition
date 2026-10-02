#include "vecadd.h"
#include "cuda_check.cuh"

__global__
void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAddDevice(const float* A_d, const float* B_d, float* C_d, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecAddKernel<<<blocks, threads>>>(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaGetLastError());
}

void vecAdd(const float* A_h, const float* B_h, float* C_h, int n) {
    size_t size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    vecAddDevice(A_d, B_d, C_d, n);

    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}