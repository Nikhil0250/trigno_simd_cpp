#include "cordic_trig.h"
#include "angle_reduction.h"   // For reduce_angle_*_SIMD
#include <immintrin.h>         // For __m256d, SIMD operations
#include <cmath>
#include <vector>
#include <algorithm>

// Precompute arctangent values for CORDIC
std::vector<double> precompute_atan(int n) {
    std::vector<double> atan_table(n);
    for (int i = 0; i < n; ++i)
        atan_table[i] = atan(pow(2.0, -i));
    return atan_table;
}

// Compute sine using CORDIC rotation mode (with SIMD-based angle reduction)
double cordic_sin(double angle) {
    __m256d ang_vec = _mm256_set1_pd(angle);
    Vec2 reduced = reduce_angle_sin_SIMD(ang_vec);

    double ang, sign;
    _mm256_storeu_pd(&ang, reduced.red);
    _mm256_storeu_pd(&sign, reduced.sign);

    std::vector<double> atan_table = precompute_atan(ITERATIONS);
    double x = 0.6072529350088812561694;
    double y = 0.0;
    double z = ang;

    for (int i = 0; i < ITERATIONS; ++i) {
        double dx = x * pow(2.0, -i);
        double dy = y * pow(2.0, -i);

        if (z >= 0) {
            double tx = x - dy;
            double ty = y + dx;
            x = tx;
            y = ty;
            z -= atan_table[i];
        } else {
            double tx = x + dy;
            double ty = y - dx;
            x = tx;
            y = ty;
            z += atan_table[i];
        }
    }
    return sign * y;
}

// Compute cosine using CORDIC rotation mode (with SIMD-based angle reduction)
double cordic_cos(double angle) {
    __m256d ang_vec = _mm256_set1_pd(angle);
    Vec2 reduced = reduce_angle_cos_SIMD(ang_vec);

    double ang, sign;
    _mm256_storeu_pd(&ang, reduced.red);
    _mm256_storeu_pd(&sign, reduced.sign);

    std::vector<double> atan_table = precompute_atan(ITERATIONS);
    double x = 0.6072529350088812561694;
    double y = 0.0;
    double z = ang;

    for (int i = 0; i < ITERATIONS; ++i) {
        double dx = x * pow(2.0, -i);
        double dy = y * pow(2.0, -i);

        if (z >= 0) {
            double tx = x - dy;
            double ty = y + dx;
            x = tx;
            y = ty;
            z -= atan_table[i];
        } else {
            double tx = x + dy;
            double ty = y - dx;
            x = tx;
            y = ty;
            z += atan_table[i];
        }
    }
    return sign * x;
}

// Compute tangent using CORDIC (with SIMD-based angle reduction)
double cordic_tan(double angle) {
    __m256d ang_vec = _mm256_set1_pd(angle);
    Vec2 reduced = reduce_angle_tan_SIMD(ang_vec);

    double ang, sign;
    _mm256_storeu_pd(&ang, reduced.red);
    _mm256_storeu_pd(&sign, reduced.sign);

    std::vector<double> atan_table = precompute_atan(ITERATIONS);
    double x = 0.6072529350088812561694;
    double y = 0.0;
    double z = ang;

    for (int i = 0; i < ITERATIONS; ++i) {
        double dx = x * pow(2.0, -i);
        double dy = y * pow(2.0, -i);

        if (z >= 0) {
            double tx = x - dy;
            double ty = y + dx;
            x = tx;
            y = ty;
            z -= atan_table[i];
        } else {
            double tx = x + dy;
            double ty = y - dx;
            x = tx;
            y = ty;
            z += atan_table[i];
        }
    }

    if (fabs(x) < 1e-12)
        return std::numeric_limits<double>::infinity();
    return sign * y / x;
}
