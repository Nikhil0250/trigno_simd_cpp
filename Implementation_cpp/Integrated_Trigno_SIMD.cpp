#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <functional>
#include <limits>
#include <utility>
#include <cstdint>
#include <immintrin.h>  // AVX2 intrinsics
#include <fstream>
#include <string>  // Needed to convert char* to std::string
// Use std namespace for brevity.
using namespace std;
using namespace std::chrono;

const double eps = 1e-12;
#define M_PID 3.14159265358979323846

struct CosHelperParams {
    double c;
    double d;
    double x;
    double y;
    double z;
};

struct SinHelperParams {
    double a;
    double b;
    double x;
    double y;
    double z;
};

struct TanHelperParams {
    double a;
    double b;
    double c;
    double d;
};

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
// Vectorized Taylor/Maclaurin Series (using 5 terms)
//=============================================================================
// Taylor series for sin(x):
// sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9!
inline __m256d taylorSin_SIMD_5(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // x
    __m256d coeff1 = _mm256_set1_pd(-0.16666666666666666);    // -1/6
    __m256d coeff2 = _mm256_set1_pd(0.008333333333333333);    // 1/120
    __m256d coeff3 = _mm256_set1_pd(-0.0001984126984126984);   // -1/5040
    __m256d coeff4 = _mm256_set1_pd(2.755731922398589e-06);    // 1/362880

    __m256d sum = _mm256_setzero_pd();
    __m256d x_power = x_vec;  // x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);
    
    // x^3 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);
    
    // x^5 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);
    
    // x^7 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);
    
    // x^9 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);
    
    return sum;
}

// Taylor series for cos(x):
// cos(x) ≈ 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320
inline __m256d taylorCos_SIMD_5(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                     // 1
    __m256d coeff1 = _mm256_set1_pd(-0.5);                      // -1/2
    __m256d coeff2 = _mm256_set1_pd(0.041666666666666666);      // 1/24
    __m256d coeff3 = _mm256_set1_pd(-0.001388888888888889);     // -1/720
    __m256d coeff4 = _mm256_set1_pd(2.48015873015873e-05);      // 1/40320

    __m256d sum = _mm256_setzero_pd();
    __m256d x_power = _mm256_set1_pd(1.0);  // x^0 = 1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);
    
    // x^2 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^2
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);
    
    // x^4 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^4
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);
    
    // x^6 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^6
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);
    
    // x^8 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^8
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);
    
    return sum;
}

// Taylor series for tan(x):
// tan(x) ≈ x + x^3/3 + 2*x^5/15 + 17*x^7/315 + 62*x^9/2835
inline __m256d taylorTan_SIMD_5(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // x
    __m256d coeff1 = _mm256_set1_pd(0.3333333333333333);      // 1/3
    __m256d coeff2 = _mm256_set1_pd(0.13333333333333333);     // 2/15
    __m256d coeff3 = _mm256_set1_pd(0.05396825396825397);      // 17/315
    __m256d coeff4 = _mm256_set1_pd(0.021869488536155202);     // 62/2835

    __m256d sum = _mm256_setzero_pd();
    __m256d x_power = x_vec;  // x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);
    
    // x^3 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);
    
    // x^5 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);
    
    // x^7 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);
    
    // x^9 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);
    
    return sum;
}

//=============================================================================
// Vectorized Taylor/Maclaurin Series (using 7 terms)
//=============================================================================
// Taylor series for sin(x) with 7 terms
// sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11! + x^13/13!
inline __m256d taylorSin_SIMD_7(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // x
    __m256d coeff1 = _mm256_set1_pd(-0.16666666666666666);    // -1/6
    __m256d coeff2 = _mm256_set1_pd(0.008333333333333333);    // 1/120
    __m256d coeff3 = _mm256_set1_pd(-0.0001984126984126984);   // -1/5040
    __m256d coeff4 = _mm256_set1_pd(2.755731922398589e-06);    // 1/362880
    __m256d coeff5 = _mm256_set1_pd(-2.505210838544172e-08);   // -1/39916800
    __m256d coeff6 = _mm256_set1_pd(1.605904383682161e-10);    // 1/6227020800

    __m256d sum = _mm256_setzero_pd();
    __m256d x_power = x_vec;  // x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);

    // x^3 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);

    // x^5 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);

    // x^7 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);

    // x^9 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);

    // x^11 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^11
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);

    // x^13 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^13
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);

    return sum;
}

// Taylor series for cos(x) with 7 terms
// cos(x) ≈ 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12!
inline __m256d taylorCos_SIMD_7(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                     // 1
    __m256d coeff1 = _mm256_set1_pd(-0.5);                      // -1/2
    __m256d coeff2 = _mm256_set1_pd(0.041666666666666666);      // 1/24
    __m256d coeff3 = _mm256_set1_pd(-0.001388888888888889);     // -1/720
    __m256d coeff4 = _mm256_set1_pd(2.48015873015873e-05);      // 1/40320
    __m256d coeff5 = _mm256_set1_pd(-2.755731922398589e-07);     // -1/3628800
    __m256d coeff6 = _mm256_set1_pd(2.08767569878681e-09);      // 1/479001600

    __m256d sum = _mm256_setzero_pd();
    __m256d x_power = _mm256_set1_pd(1.0);  // x^0 = 1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);

    // x^2 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^2
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);

    // x^4 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^4
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);

    // x^6 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^6
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);

    // x^8 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^8
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);

    // x^10 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^10
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);

    // x^12 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^12
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);

    return sum;
}

// Taylor series for tan(x) with 7 terms
// tan(x) ≈ x + x^3/3 + 2*x^5/15 + 17*x^7/315 + 62*x^9/2835 + 1382*x^11/155925 + 21844*x^13/6081075
inline __m256d taylorTan_SIMD_7(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // x
    __m256d coeff1 = _mm256_set1_pd(0.3333333333333333);      // 1/3
    __m256d coeff2 = _mm256_set1_pd(0.13333333333333333);     // 2/15
    __m256d coeff3 = _mm256_set1_pd(0.05396825396825397);     // 17/315
    __m256d coeff4 = _mm256_set1_pd(0.021869488536155202);    // 62/2835
    __m256d coeff5 = _mm256_set1_pd(0.0088632355299022);      // 1382/155925
    __m256d coeff6 = _mm256_set1_pd(0.003592128036572);       // 21844/6081075

    __m256d sum = _mm256_setzero_pd();
    __m256d x_power = x_vec;  // x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);

    // x^3 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);

    // x^5 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);

    // x^7 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);

    // x^9 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);

    // x^11 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^11
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);

    // x^13 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^13
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);

    return sum;
}

//=============================================================================
// Vectorized Taylor/Maclaurin Series (using 9 terms)
//=============================================================================
// Taylor series for sin(x) with 9 terms
// sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11! + x^13/13! - x^15/15! + x^17/17!
inline __m256d taylorSin_SIMD_9(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);
    __m256d coeff1 = _mm256_set1_pd(-0.16666666666666666);       // -1/6
    __m256d coeff2 = _mm256_set1_pd(0.008333333333333333);       // 1/120
    __m256d coeff3 = _mm256_set1_pd(-0.0001984126984126984);      // -1/5040
    __m256d coeff4 = _mm256_set1_pd(2.755731922398589e-06);       // 1/362880
    __m256d coeff5 = _mm256_set1_pd(-2.505210838544172e-08);      // -1/39916800
    __m256d coeff6 = _mm256_set1_pd(1.605904383682161e-10);       // 1/6227020800
    __m256d coeff7 = _mm256_set1_pd(-7.647163731819816e-13);      // -1/1307674368000
    __m256d coeff8 = _mm256_set1_pd(2.8114572543455206e-15);       // 1/355687428096000

    __m256d sum = _mm256_setzero_pd();
    __m256d x_power = x_vec;  // x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);

    // x^3 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);

    // x^5 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);

    // x^7 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);

    // x^9 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);

    // x^11 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^11
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);

    // x^13 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^13
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);

    // x^15 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^15
    sum = _mm256_fmadd_pd(coeff7, x_power, sum);

    // x^17 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^17
    sum = _mm256_fmadd_pd(coeff8, x_power, sum);

    return sum;
}

// Taylor series for cos(x) with 9 terms
// cos(x) ≈ 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12! - x^14/14! + x^16/16!
inline __m256d taylorCos_SIMD_9(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);
    __m256d coeff1 = _mm256_set1_pd(-0.5);
    __m256d coeff2 = _mm256_set1_pd(0.041666666666666666);
    __m256d coeff3 = _mm256_set1_pd(-0.001388888888888889);
    __m256d coeff4 = _mm256_set1_pd(2.48015873015873e-05);
    __m256d coeff5 = _mm256_set1_pd(-2.755731922398589e-07);
    __m256d coeff6 = _mm256_set1_pd(2.08767569878681e-09);
    __m256d coeff7 = _mm256_set1_pd(-1.1470745597729725e-11);
    __m256d coeff8 = _mm256_set1_pd(4.779477332387385e-13);

    __m256d sum = _mm256_setzero_pd();
    __m256d x_power = _mm256_set1_pd(1.0);  // x^0 = 1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);

    // x^2 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^2
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);

    // x^4 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^4
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);

    // x^6 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^6
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);

    // x^8 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^8
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);

    // x^10 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^10
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);

    // x^12 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^12
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);

    // x^14 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^14
    sum = _mm256_fmadd_pd(coeff7, x_power, sum);

    // x^16 term
    x_power = _mm256_mul_pd(x_power, x2);  // now x^16
    sum = _mm256_fmadd_pd(coeff8, x_power, sum);

    return sum;
}

// Taylor series for tan(x) with 9 terms
// tan(x) ≈ x + x^3/3 + 2*x^5/15 + 17*x^7/315 + 62*x^9/2835 + 1382*x^11/155925 +
//          21844*x^13/6081075 + 929569*x^15/638512875 + 6404582*x^17/10854718875
inline __m256d taylorTan_SIMD_9(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);
    __m256d coeff1 = _mm256_set1_pd(0.3333333333333333);       // 1/3
    __m256d coeff2 = _mm256_set1_pd(0.13333333333333333);      // 2/15
    __m256d coeff3 = _mm256_set1_pd(0.05396825396825397);      // 17/315
    __m256d coeff4 = _mm256_set1_pd(0.021869488536155202);     // 62/2835
    __m256d coeff5 = _mm256_set1_pd(0.0088632355299022);       // 1382/155925
    __m256d coeff6 = _mm256_set1_pd(0.003592128036572);        // 21844/6081075
    __m256d coeff7 = _mm256_set1_pd(0.001455834);              // 929569/638512875
    __m256d coeff8 = _mm256_set1_pd(0.000589041);              // 6404582/10854718875

    __m256d sum = _mm256_setzero_pd();
    __m256d x_power = x_vec;  // x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);

    // x^3 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);

    // x^5 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);

    // x^7 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);

    // x^9 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);

    // x^11 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^11
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);

    // x^13 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^13
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);

    // x^15 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^15
    sum = _mm256_fmadd_pd(coeff7, x_power, sum);

    // x^17 term
    x_power = _mm256_mul_pd(x_power, x2); // now x^17
    sum = _mm256_fmadd_pd(coeff8, x_power, sum);

    return sum;
}
//=============================================================================
// Vectorized, branchless angle reduction routines
// Each returns a pair (as a struct) of vectors: (reduced_angle, sign)
//=============================================================================
struct Vec2 {
    __m256d red;  // reduced angle
    __m256d sign; // corresponding sign (1.0 or -1.0)
};

// Compute absolute value (branchless)
inline __m256d abs256(__m256d x) {
    __m256d mask = _mm256_set1_pd(-0.0);  // sign mask: -0.0 has all bits set in the sign
    return _mm256_andnot_pd(mask, x);
}

// --- For sin(x) ---
// The original scalar routine does:
//   x = fmod(x, 2π); sign = (x<0)? -1: 1; x = |x|;
//   if (x > π) { sign = -sign; x -= π; }
//   if (x*2 >= π) { x = π - x; }
inline Vec2 reduce_angle_sin_SIMD(__m256d ang) {
    Vec2 ret;
    __m256d two_pi = _mm256_set1_pd(2 * M_PID);
    // Compute x mod 2π: x = ang - floor(ang/(2π))*(2π)
    __m256d quotient = _mm256_div_pd(ang, two_pi);
    quotient = _mm256_round_pd(quotient, _MM_FROUND_FLOOR);
    __m256d ang_mod = _mm256_sub_pd(ang, _mm256_mul_pd(quotient, two_pi));
    
    // Compute sign = (ang_mod < 0) ? -1 : 1.
    __m256d sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0),
                                    _mm256_cmp_pd(ang_mod, _mm256_set1_pd(0.0), _CMP_LT_OQ));
    // x = |ang_mod|
    __m256d x = abs256(ang_mod);
    
    // if (x > π) then sign = -sign and x = x - π.
    __m256d mask_gt_pi = _mm256_cmp_pd(x, _mm256_set1_pd(M_PID), _CMP_GT_OQ);
    sign = _mm256_mul_pd(sign,
             _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0), mask_gt_pi));
    x = _mm256_sub_pd(x, _mm256_and_pd(mask_gt_pi, _mm256_set1_pd(M_PID)));
    
    // if (x*2 >= π) then x = π - x.
    __m256d mask_ge_half_pi = _mm256_cmp_pd(_mm256_mul_pd(x, _mm256_set1_pd(2.0)),
                                             _mm256_set1_pd(M_PID), _CMP_GE_OQ);
    x = _mm256_blendv_pd(x, _mm256_sub_pd(_mm256_set1_pd(M_PID), x), mask_ge_half_pi);
    
    ret.red  = x;
    ret.sign = sign;
    return ret;
}

// --- For cos(x) ---
// The original scalar routine does:
//   x = fmod(x, 2π);
//   sign = (x > π/2 && x <= 3π/2) ? -1 : 1;
//   if (x > π) x = 2π - x; if (x > π/2) x = π - x;
inline Vec2 reduce_angle_cos_SIMD(__m256d ang) {
    Vec2 ret;
    __m256d two_pi = _mm256_set1_pd(2 * M_PID);
    __m256d quotient = _mm256_div_pd(ang, two_pi);
    quotient = _mm256_round_pd(quotient, _MM_FROUND_FLOOR);
    __m256d ang_mod = _mm256_sub_pd(ang, _mm256_mul_pd(quotient, two_pi));
    
    __m256d pi_half = _mm256_set1_pd(M_PID / 2.0);
    __m256d three_pi_half = _mm256_set1_pd(3 * M_PID / 2.0);
    // sign = -1 when (ang_mod > π/2 and ang_mod <= 3π/2)
    __m256d cmp1 = _mm256_cmp_pd(ang_mod, pi_half, _CMP_GT_OQ);
    __m256d cmp2 = _mm256_cmp_pd(ang_mod, three_pi_half, _CMP_LE_OQ);
    __m256d mask = _mm256_and_pd(cmp1, cmp2);
    __m256d sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0), mask);
    
    // if (ang_mod > π) then x = 2π - ang_mod.
    __m256d mask_gt_pi = _mm256_cmp_pd(ang_mod, _mm256_set1_pd(M_PID), _CMP_GT_OQ);
    ang_mod = _mm256_blendv_pd(ang_mod, _mm256_sub_pd(two_pi, ang_mod), mask_gt_pi);
    // if (ang_mod > π/2) then x = π - ang_mod.
    __m256d mask_gt_pi_half = _mm256_cmp_pd(ang_mod, pi_half, _CMP_GT_OQ);
    ang_mod = _mm256_blendv_pd(ang_mod, _mm256_sub_pd(_mm256_set1_pd(M_PID), ang_mod), mask_gt_pi_half);
    
    ret.red  = ang_mod;
    ret.sign = sign;
    return ret;
}

// --- For tan(x) ---
// The original scalar routine does:
//   x = fmod(x, π); if (x*2 >= π) x -= π; if (x < -π/2) x += π;
//   sign = (x < 0) ? -1 : 1; x = |x|;
inline Vec2 reduce_angle_tan_SIMD(__m256d ang) {
    Vec2 ret;
    __m256d pi = _mm256_set1_pd(M_PID);
    __m256d quotient = _mm256_div_pd(ang, pi);
    quotient = _mm256_round_pd(quotient, _MM_FROUND_FLOOR);
    __m256d ang_mod = _mm256_sub_pd(ang, _mm256_mul_pd(quotient, pi));
    
    // if (ang_mod*2 >= π) then ang_mod = ang_mod - π.
    __m256d mask = _mm256_cmp_pd(_mm256_mul_pd(ang_mod, _mm256_set1_pd(2.0)),
                                  pi, _CMP_GE_OQ);
    ang_mod = _mm256_blendv_pd(ang_mod, _mm256_sub_pd(ang_mod, pi), mask);
    
    // sign = (ang_mod < 0)? -1 : 1.
    __m256d sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0),
                                    _mm256_cmp_pd(ang_mod, _mm256_set1_pd(0.0), _CMP_LT_OQ));
    // x = |ang_mod|
    ang_mod = abs256(ang_mod);
    
    ret.red  = ang_mod;
    ret.sign = sign;
    return ret;
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
        return numeric_limits<double>::infinity();
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

double taylorSin5(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);
    Vec2 reduced = reduce_angle_sin_SIMD(ang_vec);
    __m256d sin_vec = taylorSin_SIMD_5(reduced.red);
    sin_vec = _mm256_mul_pd(sin_vec, reduced.sign);
    double result;
    _mm256_storeu_pd(&result, sin_vec);
    return result;
}

double taylorCos5(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);
    Vec2 reduced = reduce_angle_cos_SIMD(ang_vec);
    __m256d cos_vec = taylorCos_SIMD_5(reduced.red);
    cos_vec = _mm256_mul_pd(cos_vec, reduced.sign);
    double result;
    _mm256_storeu_pd(&result, cos_vec);
    return result;
}

double maclaurinTan5(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);
    Vec2 reduced = reduce_angle_tan_SIMD(ang_vec);
    __m256d tan_vec = taylorTan_SIMD_5(reduced.red);
    tan_vec = _mm256_mul_pd(tan_vec, reduced.sign);
    double result;
    _mm256_storeu_pd(&result, tan_vec);
    return result;
}


double taylorSin7(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);
    Vec2 reduced = reduce_angle_sin_SIMD(ang_vec);
    __m256d sin_vec = taylorSin_SIMD_7(reduced.red);
    sin_vec = _mm256_mul_pd(sin_vec, reduced.sign);
    double result;
    _mm256_storeu_pd(&result, sin_vec);
    return result;
}

double taylorCos7(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);
    Vec2 reduced = reduce_angle_cos_SIMD(ang_vec);
    __m256d cos_vec = taylorCos_SIMD_7(reduced.red);
    cos_vec = _mm256_mul_pd(cos_vec, reduced.sign);
    double result;
    _mm256_storeu_pd(&result, cos_vec);
    return result;
}

double maclaurinTan7(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);
    Vec2 reduced = reduce_angle_tan_SIMD(ang_vec);
    __m256d tan_vec = taylorTan_SIMD_7(reduced.red);
    tan_vec = _mm256_mul_pd(tan_vec, reduced.sign);
    double result;
    _mm256_storeu_pd(&result, tan_vec);
    return result;
}


double taylorSin9(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);
    Vec2 reduced = reduce_angle_sin_SIMD(ang_vec);
    __m256d sin_vec = taylorSin_SIMD_9(reduced.red);
    sin_vec = _mm256_mul_pd(sin_vec, reduced.sign);
    double result;
    _mm256_storeu_pd(&result, sin_vec);
    return result;
}

double taylorCos9(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);
    Vec2 reduced = reduce_angle_cos_SIMD(ang_vec);
    __m256d cos_vec = taylorCos_SIMD_9(reduced.red);
    cos_vec = _mm256_mul_pd(cos_vec, reduced.sign);
    double result;
    _mm256_storeu_pd(&result, cos_vec);
    return result;
}

double maclaurinTan9(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);
    Vec2 reduced = reduce_angle_tan_SIMD(ang_vec);
    __m256d tan_vec = taylorTan_SIMD_9(reduced.red);
    tan_vec = _mm256_mul_pd(tan_vec, reduced.sign);
    double result;
    _mm256_storeu_pd(&result, tan_vec);
    return result;
}

//=============================================================================
// Benchmarking Infrastructure
//=============================================================================
using FuncPtr = double(*)(double);

struct BenchmarkResult {
    double angle;
    double std_result;
    double my_result;
    double abs_error;
    double rel_error; // When std_result is near zero, this is set to 0.
    long long std_time_ns; // Execution time for built-in function.
    long long my_time_ns;  // Execution time for your implementation.
};

struct Bench_Response {
    double abs_err_max;
    double abs_err_avg;
    double abs_std_dev;
    double rel_err_max;
    double rel_err_avg;
    double rel_std_dev;
    long long std_time_total;
    long long my_func_total;
    int wins;
    int losses;
    int equals;
};

void save_performance_data(const std::vector<BenchmarkResult>& results, const char* method_name) {
    std::string filename = std::string(method_name) + "_performance.csv";  
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header row
    file << "angle,std_result,my_result,abs_error,rel_error,std_time,my_time,diff,speedup_ratio\n";

    // Write data rows
    for (const auto& r : results) {
        long long diff = r.std_time_ns - r.my_time_ns;
        double speedup_ratio;

        if (r.std_time_ns == 0 && r.my_time_ns == 0) {
            speedup_ratio = 1.0;  // No difference in execution
        } else if (r.my_time_ns == 0) {
            speedup_ratio = std::numeric_limits<double>::infinity();  // My function is extremely fast
        } else if (r.std_time_ns == 0) {
            speedup_ratio = 0.0;  // My function is infinitely slower
        } else {
            speedup_ratio = static_cast<double>(r.std_time_ns) / r.my_time_ns;
        }

        file << r.angle << "," 
             << r.std_result << "," 
             << r.my_result << "," 
             << r.abs_error << "," 
             << r.rel_error << "," 
             << r.std_time_ns << "," 
             << r.my_time_ns << "," 
             << diff << ","
             << speedup_ratio << "\n";
    }
    
    file.close();
    std::cout << "Saved performance data for " << method_name << " to " << filename << std::endl;
}


template<typename Func>
long long measure_execution_time(Func f, double input, size_t iterations) {
    volatile double res;
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i)
        res = f(input);
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count();
}

Bench_Response run_benchmark(const char* func_name, FuncPtr std_func, FuncPtr my_func,
                             size_t num_angles = 1000, size_t iterations = 100000) {
    vector<BenchmarkResult> results;
    results.reserve(num_angles);

    mt19937_64 rng(2024873);
    uniform_real_distribution<double> angle_dist(-1e6, 1e6);
    const double epsilon = 1e-12;

    for (size_t i = 0; i < num_angles; ++i) {
        double angle = angle_dist(rng);
        double std_val = std_func(angle);
        double my_val  = my_func(angle);
        double abs_err = fabs(my_val - std_val);
        double rel_err = (fabs(std_val) > epsilon) ? abs_err / fabs(std_val) : 0.0;
        long long std_time = measure_execution_time(std_func, angle, iterations);
        long long my_time  = measure_execution_time(my_func, angle, iterations);
        results.push_back({angle, std_val, my_val, abs_err, rel_err, std_time, my_time});
    }
    Bench_Response Local_Entry = {};
    double sum_abs = 0.0, sum_rel = 0.0;
    double max_abs = 0.0, max_rel = 0.0;
    vector<double> abs_errs, rel_errs;
    long long total_std_time = 0, total_my_time = 0;
    int losses = 0, wins = 0, equals = 0;
    
    for (const auto &r : results) {
        if (r.my_time_ns > r.std_time_ns){
            losses++;
        }else if (r.my_time_ns < r.std_time_ns){
            wins++;
        }else{
            equals++;
        }
        sum_abs += r.abs_error;
        sum_rel += r.rel_error;
        if (r.abs_error > max_abs)
            max_abs = r.abs_error;
        if (r.rel_error > max_rel)
            max_rel = r.rel_error;
        abs_errs.push_back(r.abs_error);
        rel_errs.push_back(r.rel_error);
        total_std_time += r.std_time_ns;
        total_my_time  += r.my_time_ns;
    }
    
    size_t result_size = results.size();
    double avg_abs = sum_abs / result_size;
    double avg_rel = sum_rel / result_size;
    double stddev_abs = 0.0, stddev_rel = 0.0;
    for (auto e : abs_errs)
        stddev_abs += (e - avg_abs) * (e - avg_abs);
    for (auto e : rel_errs)
        stddev_rel += (e - avg_rel) * (e - avg_rel);
    stddev_abs = sqrt(stddev_abs / result_size);
    stddev_rel = sqrt(stddev_rel / result_size);

    Local_Entry.abs_err_max = max_abs;
    Local_Entry.abs_err_avg = avg_abs;
    Local_Entry.abs_std_dev = stddev_abs;
    Local_Entry.rel_err_max = max_rel;
    Local_Entry.rel_err_avg = avg_rel;
    Local_Entry.rel_std_dev = stddev_rel;
    Local_Entry.std_time_total = total_std_time;
    Local_Entry.my_func_total = total_my_time;
    Local_Entry.wins = wins;
    Local_Entry.losses = losses;
    Local_Entry.equals = equals;

    // save_performance_data(results, func_name); // to save data to csv file for making graphs in python

    return Local_Entry;
}

struct Final_Total_Stats {
    double total_abs_err_max = 0.0;
    double total_abs_err_avg = 0.0;
    double total_abs_std_dev = 0.0;
    double total_rel_err_max = 0.0;
    double total_rel_err_avg = 0.0;
    double total_rel_std_dev = 0.0;
    long long total_std_time_total = 0;
    long long total_my_func_total = 0;
    double total_wins = 0.0;
    double total_losses = 0.0;
    double total_equals = 0.0;
};

void updateStats(Final_Total_Stats& total_stats, const Bench_Response& new_data) {
    total_stats.total_abs_err_max += new_data.abs_err_max;
    total_stats.total_abs_err_avg += new_data.abs_err_avg;
    total_stats.total_abs_std_dev += new_data.abs_std_dev;
    total_stats.total_rel_err_max += new_data.rel_err_max;
    total_stats.total_rel_err_avg += new_data.rel_err_avg;
    total_stats.total_rel_std_dev += new_data.rel_std_dev;
    total_stats.total_std_time_total += new_data.std_time_total;
    total_stats.total_my_func_total += new_data.my_func_total;
    total_stats.total_wins += new_data.wins;
    total_stats.total_losses += new_data.losses;
    total_stats.total_equals += new_data.equals;
}

void printStats(const Final_Total_Stats& stats, const string& label) {
    cout << "==============================================" << endl;
    cout << "         AVERAGE " << label << " STATISTICS" << endl;
    cout << "==============================================" << endl;
    cout << fixed << setprecision(6);
    cout << left << setw(45) << "Average Absolute Max Error: " << stats.total_abs_err_max << endl;
    cout << left << setw(45) << "Average Absolute Error: " << stats.total_abs_err_avg << endl;
    cout << left << setw(45) << "Average Absolute Std Dev: " << stats.total_abs_std_dev << endl;
    cout << left << setw(45) << "Average Relative Max Error: " << stats.total_rel_err_max << endl;
    cout << left << setw(45) << "Average Relative Error: " << stats.total_rel_err_avg << endl;
    cout << left << setw(45) << "Average Relative Std Dev: " << stats.total_rel_std_dev << endl;
    cout << left << setw(45) << "Average Standard Time: " << stats.total_std_time_total << endl;
    cout << left << setw(45) << "Average My Function Time: " << stats.total_my_func_total << endl;
    cout << left << setw(45) << "Average Wins: " << stats.total_wins << endl;
    cout << left << setw(45) << "Average Losses: " << stats.total_losses << endl;
    cout << left << setw(45) << "Average Equals: " << stats.total_equals << endl;
    cout << "==============================================" << endl;
}

//=============================================================================
// Main: Run Benchmarks for sin, cos, and tan (and their Taylor variants)
//=============================================================================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    Final_Total_Stats sin_stats;
    Final_Total_Stats cos_stats;
    Final_Total_Stats tan_stats;
    Final_Total_Stats taylor_sin5_stats;
    Final_Total_Stats taylor_cos5_stats;
    Final_Total_Stats taylor_tan5_stats;
    Final_Total_Stats taylor_sin7_stats;
    Final_Total_Stats taylor_cos7_stats;
    Final_Total_Stats taylor_tan7_stats;
    Final_Total_Stats taylor_sin9_stats;
    Final_Total_Stats taylor_cos9_stats;
    Final_Total_Stats taylor_tan9_stats;
    
        Bench_Response sin_bench = run_benchmark("mysin", sin, my_sin);
        updateStats(sin_stats, sin_bench);
        Bench_Response cos_bench = run_benchmark("mycos", cos, my_cos);
        updateStats(cos_stats, cos_bench);
        Bench_Response tan_bench = run_benchmark("mytan", tan, my_tan);
        updateStats(tan_stats, tan_bench);
        Bench_Response taylor_sin5_bench = run_benchmark("taylorsin5", sin, taylorSin5);
        updateStats(taylor_sin5_stats, taylor_sin5_bench);
        Bench_Response taylor_cos5_bench = run_benchmark("taylorcos5", cos, taylorCos5);
        updateStats(taylor_cos5_stats, taylor_cos5_bench);
        Bench_Response taylor_tan5_bench = run_benchmark("taylortan5", tan, maclaurinTan5);
        updateStats(taylor_tan5_stats, taylor_tan5_bench);
        Bench_Response taylor_sin7_bench = run_benchmark("taylorsin7", sin, taylorSin7);
        updateStats(taylor_sin7_stats, taylor_sin7_bench);
        Bench_Response taylor_cos7_bench = run_benchmark("taylorcos7", cos, taylorCos7);
        updateStats(taylor_cos7_stats, taylor_cos7_bench);
        Bench_Response taylor_tan7_bench = run_benchmark("taylortan7", tan, maclaurinTan7);
        updateStats(taylor_tan7_stats, taylor_tan7_bench);
        Bench_Response taylor_sin9_bench = run_benchmark("taylorsin9", sin, taylorSin9);
        updateStats(taylor_sin9_stats, taylor_sin9_bench);
        Bench_Response taylor_cos9_bench = run_benchmark("taylorcos9", cos, taylorCos9);
        updateStats(taylor_cos9_stats, taylor_cos9_bench);
        Bench_Response taylor_tan9_bench = run_benchmark("taylortan9", tan, maclaurinTan9);
        updateStats(taylor_tan9_stats, taylor_tan9_bench);
    
     
    printStats(sin_stats, "SINE");
    printStats(cos_stats, "COSINE");
    printStats(tan_stats, "TANGENT");
    printStats(taylor_sin5_stats, "TAYLOR 5 SINE");
    printStats(taylor_cos5_stats, "TAYLOR 5 COSINE");
    printStats(taylor_tan5_stats, "TAYLOR 5 TANGENT");
    printStats(taylor_sin7_stats, "TAYLOR 7 SINE");
    printStats(taylor_cos7_stats, "TAYLOR 7 COSINE");
    printStats(taylor_tan7_stats, "TAYLOR 7 TANGENT");
    printStats(taylor_sin9_stats, "TAYLOR 9 SINE");
    printStats(taylor_cos9_stats, "TAYLOR 9 COSINE");
    printStats(taylor_tan9_stats, "TAYLOR 9 TANGENT");

    
    return 0;
}
