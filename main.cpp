#include "vecadd.h"
#include "vecadd_cpu.h"
#include "kernel_benchmark.h"
#include "timer.h"
#include <cstdio>
#include <vector>

int main() {
    int repeats = 20;
    std::vector<float> warmup(1, 0.0f);
    vecAdd(warmup.data(), warmup.data(), warmup.data(), 1);

    printf("%12s %12s %16s %16s\n", "N", "CPU (ms)", "GPU total (ms)", "GPU kernel (ms)");

    for (int shift = 10; shift <= 26; shift += 4) {
        int n = 1 << shift;
        std::vector<float> A(n, 1.0f), B(n, 2.0f), C(n);

        double cpuMs = averageMs([&] { vecAddCpu(A.data(), B.data(), C.data(), n); }, repeats);
        double gpuTotalMs = averageMs([&] { vecAdd(A.data(), B.data(), C.data(), n); }, repeats);
        double gpuKernelMs = vecAddKernelMs(n, repeats);

        printf("%12d %12.4f %16.4f %16.4f\n", n, cpuMs, gpuTotalMs, gpuKernelMs);
    }
}