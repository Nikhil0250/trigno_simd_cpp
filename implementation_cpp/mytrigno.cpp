#include "mytrigno.h"
#include<limits>
#include <immintrin.h> // For AVX intrinsics

SinHelperParams sinparams[7] = {
    {630.2535746, 0.0, 400502.374645736, -72421.86530064, 399424.0},  // Parameters for range [0, intervalWidth)
    {572.9577951, 10.0, 380805.536587892, -289687.4612814 , 431749.0},  // Parameters for range [intervalWidth, 2*intervalWidth)
    {343.7746771, 46.0, 200251.187385321, -278342.89691 ,294797.0},  // Parameters for range [2*intervalWidth, 3*intervalWidth)
    {572.9577951, 217.0, 883074.908162424, -1616428.5317114 , 1614593.0},  // Parameters for range [3*intervalWidth, 4*intervalWidth)
    {229.1831181, 297.0, 380805.536587892,-906648.4149306 , 916309.0},  // Parameters for range [4*intervalWidth, 5*intervalWidth)
    {57.29577951, 542.0, 400502.374645736, -1185793.45271916, 1273864.0},   // Parameters for range [5*intervalWidth, RANGE_MAX] (or above)
    {57.29577951, 542.0, 400502.374645736, -1185793.45271916, 1273864.0}   // Out of bounds Handling
};

CosHelperParams cosparams[7] = {
    {-57.29577951, 632.0, 400502.374645736, -72421.86530064, 399424.0},  // Parameters for range [0, intervalWidth)
    {-229.1831181, 657.0, 380805.536587892, -289687.4612814 , 431749.0},  // Parameters for range [intervalWidth, 2*intervalWidth)
    {-286.4788976, 541.0, 200251.187385321, -278342.89691 ,294797.0},  // Parameters for range [2*intervalWidth, 3*intervalWidth)
    {-744.8451337, 1252.0, 883074.908162424, -1616428.5317114 , 1614593.0},  // Parameters for range [3*intervalWidth, 4*intervalWidth)
    {-572.9577951, 910.0, 380805.536587892,-906648.4149306 , 916309.0},  // Parameters for range [4*intervalWidth, 5*intervalWidth)
    {-630.2535746, 990.0, 400502.374645736, -1185793.45271916, 1273864.0} ,  // Parameters for range [5*intervalWidth, RANGE_MAX] (or above)
    {-630.2535746, 990.0, 400502.374645736, -1185793.45271916, 1273864.0}   // Out of Bounds Handling
};

TanHelperParams tanparams[7] = {
    {630.2535746, 0.0, -57.29577951, 632.0},  // Parameters for range [0, intervalWidth)
    {572.9577951, 10.0, -229.1831181, 657.0},  // Parameters for range [intervalWidth, 2*intervalWidth)
    {343.7746771, 46.0, -286.4788976, 541.0},  // Parameters for range [2*intervalWidth, 3*intervalWidth)
    {572.9577951, 217.0, -744.8451337, 1252.0},  // Parameters for range [3*intervalWidth, 4*intervalWidth)
    {229.1831181, 297.0, -572.9577951, 910.0},  // Parameters for range [4*intervalWidth, 5*intervalWidth)
    {57.29577951, 542.0, -630.2535746, 990.0},   // Parameters for range [5*intervalWidth, RANGE_MAX] (or above)
    {57.29577951, 542.0, -630.2535746, 990.0} //out of bounds 
};

//=============================================================================
// SIMD Helpers
//=============================================================================

// Fast inverse square-root (applied element‑wise) using AVX2.
// (This uses the “magic number” method plus one iteration of Newton–Raphson.)
inline __m256d fastInverseSqrt_SIMD(__m256d number) {
    const __m256d threeHalfs = _mm256_set1_pd(1.5);
    __m256d x2 = _mm256_mul_pd(number, _mm256_set1_pd(0.5));
    
    // Reinterpret the double vector as 64-bit integers.
    __m256i i = _mm256_castpd_si256(number);
    i = _mm256_sub_epi64(_mm256_set1_epi64x(0x5fe6eb50c7b537a9), _mm256_srli_epi64(i, 1));
    __m256d y = _mm256_castsi256_pd(i);
    
    // One iteration of Newton–Raphson:
    y = _mm256_mul_pd(y, _mm256_sub_pd(threeHalfs,
          _mm256_mul_pd(x2, _mm256_mul_pd(y, y))));
    return y;
}

//=============================================================================
// Fused Multiply–Add based helper functions
//=============================================================================

inline __m256d sin_helper_SIMD(__m256d a, __m256d b,
    __m256d x, __m256d y, __m256d z,
    __m256d ang) {
// Compute: (a*ang + b) * fastInverseSqrt(x*ang^2 + y*ang + z)
__m256d poly  = _mm256_fmadd_pd(a, ang, b);
__m256d ang2  = _mm256_mul_pd(ang, ang);
__m256d inner = _mm256_fmadd_pd(x, ang2, _mm256_fmadd_pd(y, ang, z));
__m256d invSqrt = fastInverseSqrt_SIMD(inner);
return _mm256_mul_pd(poly, invSqrt);
}

inline __m256d cos_helper_SIMD(__m256d c, __m256d d,
    __m256d x, __m256d y, __m256d z,
    __m256d ang) {
// Compute: (c*ang + d) * fastInverseSqrt(x*ang^2 + y*ang + z)
__m256d poly  = _mm256_fmadd_pd(c, ang, d);
__m256d ang2  = _mm256_mul_pd(ang, ang);
__m256d inner = _mm256_fmadd_pd(x, ang2, _mm256_fmadd_pd(y, ang, z));
__m256d invSqrt = fastInverseSqrt_SIMD(inner);
return _mm256_mul_pd(poly, invSqrt);
}

inline __m256d tan_helper_SIMD(__m256d a, __m256d b,
    __m256d c, __m256d d,
    __m256d ang) {
// Compute: ((a*ang)+b) * [fastInverseSqrt(c*ang+d)]^2
__m256d poly = _mm256_fmadd_pd(a, ang, b);
__m256d tmp  = fastInverseSqrt_SIMD(_mm256_fmadd_pd(c, ang, d));
return _mm256_mul_pd(poly, _mm256_mul_pd(tmp, tmp));
}

//=============================================================================
// Exposed Functions (scalar API that internally uses SIMD)
//=============================================================================

double my_sin(double ang) {
    // Broadcast the scalar angle into all 4 lanes.
    __m256d ang_vec = _mm256_set1_pd(ang);
    // Use the SIMD branchless angle reduction for sin.
    Vec2 reduced = reduce_angle_sin_SIMD(ang_vec);
    // Extract one lane to decide on the parameter index.
    double red_scalar;
    _mm256_storeu_pd(&red_scalar, reduced.red);  // only the first element is used
    int idx = static_cast<int>(red_scalar / 0.261);
    SinHelperParams p = sinparams[idx];
    
    __m256d a = _mm256_set1_pd(p.a);
    __m256d b = _mm256_set1_pd(p.b);
    __m256d x = _mm256_set1_pd(p.x);
    __m256d y = _mm256_set1_pd(p.y);
    __m256d z = _mm256_set1_pd(p.z);
    
    __m256d res_vec = sin_helper_SIMD(a, b, x, y, z, reduced.red);
    res_vec = _mm256_mul_pd(res_vec, reduced.sign);
    
    double result;
    _mm256_storeu_pd(&result, res_vec);
    return result;
}

double my_cos(double ang) {
    __m256d ang_vec = _mm256_set1_pd(ang);
    Vec2 reduced = reduce_angle_cos_SIMD(ang_vec);
    double red_scalar;
    _mm256_storeu_pd(&red_scalar, reduced.red);
    int idx = static_cast<int>(red_scalar / 0.261);
    CosHelperParams p = cosparams[idx];
    
    __m256d c = _mm256_set1_pd(p.c);
    __m256d d = _mm256_set1_pd(p.d);
    __m256d x = _mm256_set1_pd(p.x);
    __m256d y = _mm256_set1_pd(p.y);
    __m256d z = _mm256_set1_pd(p.z);
    
    __m256d res_vec = cos_helper_SIMD(c, d, x, y, z, reduced.red);
    res_vec = _mm256_mul_pd(res_vec, reduced.sign);
    
    double result;
    _mm256_storeu_pd(&result, res_vec);
    return result;
}

double my_tan(double ang) {
    __m256d ang_vec = _mm256_set1_pd(ang);
    Vec2 reduced = reduce_angle_tan_SIMD(ang_vec);
    double red_scalar;
    _mm256_storeu_pd(&red_scalar, reduced.red);
    // Check for near π/2 (avoid division-by-zero issues)
    if (fabs(red_scalar - M_PID / 2.0) < 1e-3)
        return std::numeric_limits<double>::infinity();
    int idx = static_cast<int>(red_scalar / 0.261);
    TanHelperParams p = tanparams[idx];
    
    __m256d a = _mm256_set1_pd(p.a);
    __m256d b = _mm256_set1_pd(p.b);
    __m256d c = _mm256_set1_pd(p.c);
    __m256d d = _mm256_set1_pd(p.d);
    
    __m256d res_vec = tan_helper_SIMD(a, b, c, d, reduced.red);
    res_vec = _mm256_mul_pd(res_vec, reduced.sign);
    
    double result;
    _mm256_storeu_pd(&result, res_vec);
    return result;
}
