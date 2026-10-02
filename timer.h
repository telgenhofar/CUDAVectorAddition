#pragma once
#include <chrono>

template <typename F>
double averageMs(F work, int repeats) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; i++) {
        work();
    }
    std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
    return elapsed.count() / repeats;
}