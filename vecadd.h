#pragma once

void vecAdd(const float* A_h, const float* B_h, float* C_h, int n);\
void vecAddDevice(const float* A_d, const float* B_d, float* C_d, int n);