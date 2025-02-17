#include "taylorsimd.h"

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
