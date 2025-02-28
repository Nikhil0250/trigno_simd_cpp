#ifndef TAYLORSIMD_H
#define TAYLORSIMD_H

#include <immintrin.h>  // AVX intrinsics
#include "angle_reduction.h"  // Required for reduce_angle functions

// Taylor series function declarations
__m256d taylorSin_SIMD_5(__m256d x_vec);
__m256d taylorCos_SIMD_5(__m256d x_vec);
__m256d taylorTan_SIMD_5(__m256d x_vec);

__m256d taylorSin_SIMD_7(__m256d x_vec);
__m256d taylorCos_SIMD_7(__m256d x_vec);
__m256d taylorTan_SIMD_7(__m256d x_vec);

__m256d taylorSin_SIMD_9(__m256d x_vec);
__m256d taylorCos_SIMD_9(__m256d x_vec);
__m256d taylorTan_SIMD_9(__m256d x_vec);

// Scalar wrapper functions
double taylorSin5(double x);
double taylorCos5(double x);
double maclaurinTan5(double x);

double taylorSin7(double x);
double taylorCos7(double x);
double maclaurinTan7(double x);

double taylorSin9(double x);
double taylorCos9(double x);
double maclaurinTan9(double x);

#endif  // TAYLORSIMD_H
