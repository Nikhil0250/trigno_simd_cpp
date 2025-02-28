// Include the corresponding header file for function declarations.
// Ensures consistency between function definitions and their declarations.
#include "taylorsimd.h"

//=============================================================================
// Vectorized Taylor/Maclaurin Series (using 5 terms)
//=============================================================================

// Inline function to compute sine using a 5-term Taylor series with SIMD instructions.
// Approximates sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! for a 256-bit vector of angles.
inline __m256d taylorSin_SIMD_5(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // Coefficient for x^1 term: 1
    __m256d coeff1 = _mm256_set1_pd(-0.16666666666666666);    // Coefficient for x^3 term: -1/6
    __m256d coeff2 = _mm256_set1_pd(0.008333333333333333);    // Coefficient for x^5 term: 1/120
    __m256d coeff3 = _mm256_set1_pd(-0.0001984126984126984);   // Coefficient for x^7 term: -1/5040
    __m256d coeff4 = _mm256_set1_pd(2.755731922398589e-06);    // Coefficient for x^9 term: 1/362880

    __m256d sum = _mm256_setzero_pd();  // Initialize sum to zero
    __m256d x_power = x_vec;            // Start with x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);  // Add x^1 term
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);     // Compute x^2 for successive powers
    
    // Compute x^3 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);  // Add x^3 term
    
    // Compute x^5 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);  // Add x^5 term
    
    // Compute x^7 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);  // Add x^7 term
    
    // Compute x^9 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);  // Add x^9 term
    
    return sum;  // Return the 5-term sine approximation
}

// Inline function to compute cosine using a 5-term Taylor series with SIMD instructions.
// Approximates cos(x) ≈ 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! for a 256-bit vector of angles.
inline __m256d taylorCos_SIMD_5(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                     // Coefficient for x^0 term: 1
    __m256d coeff1 = _mm256_set1_pd(-0.5);                      // Coefficient for x^2 term: -1/2
    __m256d coeff2 = _mm256_set1_pd(0.041666666666666666);      // Coefficient for x^4 term: 1/24
    __m256d coeff3 = _mm256_set1_pd(-0.001388888888888889);     // Coefficient for x^6 term: -1/720
    __m256d coeff4 = _mm256_set1_pd(2.48015873015873e-05);      // Coefficient for x^8 term: 1/40320

    __m256d sum = _mm256_setzero_pd();        // Initialize sum to zero
    __m256d x_power = _mm256_set1_pd(1.0);    // Start with x^0 = 1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);  // Add x^0 term
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);     // Compute x^2 for successive powers
    
    // Compute x^2 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^2
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);  // Add x^2 term
    
    // Compute x^4 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^4
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);  // Add x^4 term
    
    // Compute x^6 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^6
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);  // Add x^6 term
    
    // Compute x^8 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^8
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);  // Add x^8 term
    
    return sum;  // Return the 5-term cosine approximation
}

// Inline function to compute tangent using a 5-term Maclaurin series with SIMD instructions.
// Approximates tan(x) ≈ x + x^3/3 + 2*x^5/15 + 17*x^7/315 + 62*x^9/2835 for a 256-bit vector of angles.
inline __m256d taylorTan_SIMD_5(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // Coefficient for x^1 term: 1
    __m256d coeff1 = _mm256_set1_pd(0.3333333333333333);      // Coefficient for x^3 term: 1/3
    __m256d coeff2 = _mm256_set1_pd(0.13333333333333333);     // Coefficient for x^5 term: 2/15
    __m256d coeff3 = _mm256_set1_pd(0.05396825396825397);      // Coefficient for x^7 term: 17/315
    __m256d coeff4 = _mm256_set1_pd(0.021869488536155202);     // Coefficient for x^9 term: 62/2835

    __m256d sum = _mm256_setzero_pd();  // Initialize sum to zero
    __m256d x_power = x_vec;            // Start with x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);  // Add x^1 term
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);     // Compute x^2 for successive powers
    
    // Compute x^3 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);  // Add x^3 term
    
    // Compute x^5 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);  // Add x^5 term
    
    // Compute x^7 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);  // Add x^7 term
    
    // Compute x^9 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);  // Add x^9 term
    
    return sum;  // Return the 5-term tangent approximation
}

//=============================================================================
// Vectorized Taylor/Maclaurin Series (using 7 terms)
//=============================================================================

// Inline function to compute sine using a 7-term Taylor series with SIMD instructions.
// Approximates sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11! + x^13/13! for a 256-bit vector.
inline __m256d taylorSin_SIMD_7(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // Coefficient for x^1 term: 1
    __m256d coeff1 = _mm256_set1_pd(-0.16666666666666666);    // Coefficient for x^3 term: -1/6
    __m256d coeff2 = _mm256_set1_pd(0.008333333333333333);    // Coefficient for x^5 term: 1/120
    __m256d coeff3 = _mm256_set1_pd(-0.0001984126984126984);   // Coefficient for x^7 term: -1/5040
    __m256d coeff4 = _mm256_set1_pd(2.755731922398589e-06);    // Coefficient for x^9 term: 1/362880
    __m256d coeff5 = _mm256_set1_pd(-2.505210838544172e-08);   // Coefficient for x^11 term: -1/39916800
    __m256d coeff6 = _mm256_set1_pd(1.605904383682161e-10);    // Coefficient for x^13 term: 1/6227020800

    __m256d sum = _mm256_setzero_pd();  // Initialize sum to zero
    __m256d x_power = x_vec;            // Start with x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);  // Add x^1 term
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);     // Compute x^2 for successive powers

    // Compute x^3 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);  // Add x^3 term

    // Compute x^5 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);  // Add x^5 term

    // Compute x^7 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);  // Add x^7 term

    // Compute x^9 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);  // Add x^9 term

    // Compute x^11 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^11
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);  // Add x^11 term

    // Compute x^13 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^13
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);  // Add x^13 term

    return sum;  // Return the 7-term sine approximation
}

// Inline function to compute cosine using a 7-term Taylor series with SIMD instructions.
// Approximates cos(x) ≈ 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12! for a 256-bit vector.
inline __m256d taylorCos_SIMD_7(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                     // Coefficient for x^0 term: 1
    __m256d coeff1 = _mm256_set1_pd(-0.5);                      // Coefficient for x^2 term: -1/2
    __m256d coeff2 = _mm256_set1_pd(0.041666666666666666);      // Coefficient for x^4 term: 1/24
    __m256d coeff3 = _mm256_set1_pd(-0.001388888888888889);     // Coefficient for x^6 term: -1/720
    __m256d coeff4 = _mm256_set1_pd(2.48015873015873e-05);      // Coefficient for x^8 term: 1/40320
    __m256d coeff5 = _mm256_set1_pd(-2.755731922398589e-07);     // Coefficient for x^10 term: -1/3628800
    __m256d coeff6 = _mm256_set1_pd(2.08767569878681e-09);      // Coefficient for x^12 term: 1/479001600

    __m256d sum = _mm256_setzero_pd();        // Initialize sum to zero
    __m256d x_power = _mm256_set1_pd(1.0);    // Start with x^0 = 1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);  // Add x^0 term
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);     // Compute x^2 for successive powers

    // Compute x^2 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^2
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);  // Add x^2 term

    // Compute x^4 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^4
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);  // Add x^4 term

    // Compute x^6 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^6
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);  // Add x^6 term

    // Compute x^8 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^8
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);  // Add x^8 term

    // Compute x^10 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^10
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);  // Add x^10 term

    // Compute x^12 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^12
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);  // Add x^12 term

    return sum;  // Return the 7-term cosine approximation
}

// Inline function to compute tangent using a 7-term Maclaurin series with SIMD instructions.
// Approximates tan(x) ≈ x + x^3/3 + 2*x^5/15 + 17*x^7/315 + 62*x^9/2835 + 1382*x^11/155925 + 21844*x^13/6081075.
inline __m256d taylorTan_SIMD_7(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // Coefficient for x^1 term: 1
    __m256d coeff1 = _mm256_set1_pd(0.3333333333333333);      // Coefficient for x^3 term: 1/3
    __m256d coeff2 = _mm256_set1_pd(0.13333333333333333);     // Coefficient for x^5 term: 2/15
    __m256d coeff3 = _mm256_set1_pd(0.05396825396825397);     // Coefficient for x^7 term: 17/315
    __m256d coeff4 = _mm256_set1_pd(0.021869488536155202);    // Coefficient for x^9 term: 62/2835
    __m256d coeff5 = _mm256_set1_pd(0.0088632355299022);      // Coefficient for x^11 term: 1382/155925
    __m256d coeff6 = _mm256_set1_pd(0.003592128036572);       // Coefficient for x^13 term: 21844/6081075

    __m256d sum = _mm256_setzero_pd();  // Initialize sum to zero
    __m256d x_power = x_vec;            // Start with x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);  // Add x^1 term
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);     // Compute x^2 for successive powers

    // Compute x^3 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);  // Add x^3 term

    // Compute x^5 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);  // Add x^5 term

    // Compute x^7 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);  // Add x^7 term

    // Compute x^9 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);  // Add x^9 term

    // Compute x^11 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^11
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);  // Add x^11 term

    // Compute x^13 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^13
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);  // Add x^13 term

    return sum;  // Return the 7-term tangent approximation
}

//=============================================================================
// Vectorized Taylor/Maclaurin Series (using 9 terms)
//=============================================================================

// Inline function to compute sine using a 9-term Taylor series with SIMD instructions.
// Approximates sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11! + x^13/13! - x^15/15! + x^17/17!.
inline __m256d taylorSin_SIMD_9(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // Coefficient for x^1 term: 1
    __m256d coeff1 = _mm256_set1_pd(-0.16666666666666666);    // Coefficient for x^3 term: -1/6
    __m256d coeff2 = _mm256_set1_pd(0.008333333333333333);    // Coefficient for x^5 term: 1/120
    __m256d coeff3 = _mm256_set1_pd(-0.0001984126984126984);   // Coefficient for x^7 term: -1/5040
    __m256d coeff4 = _mm256_set1_pd(2.755731922398589e-06);    // Coefficient for x^9 term: 1/362880
    __m256d coeff5 = _mm256_set1_pd(-2.505210838544172e-08);   // Coefficient for x^11 term: -1/39916800
    __m256d coeff6 = _mm256_set1_pd(1.605904383682161e-10);    // Coefficient for x^13 term: 1/6227020800
    __m256d coeff7 = _mm256_set1_pd(-7.647163731819816e-13);   // Coefficient for x^15 term: -1/1307674368000
    __m256d coeff8 = _mm256_set1_pd(2.8114572543455206e-15);   // Coefficient for x^17 term: 1/355687428096000

    __m256d sum = _mm256_setzero_pd();  // Initialize sum to zero
    __m256d x_power = x_vec;            // Start with x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);  // Add x^1 term
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);     // Compute x^2 for successive powers

    // Compute x^3 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);  // Add x^3 term

    // Compute x^5 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);  // Add x^5 term

    // Compute x^7 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);  // Add x^7 term

    // Compute x^9 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);  // Add x^9 term

    // Compute x^11 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^11
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);  // Add x^11 term

    // Compute x^13 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^13
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);  // Add x^13 term

    // Compute x^15 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^15
    sum = _mm256_fmadd_pd(coeff7, x_power, sum);  // Add x^15 term

    // Compute x^17 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^17
    sum = _mm256_fmadd_pd(coeff8, x_power, sum);  // Add x^17 term

    return sum;  // Return the 9-term sine approximation
}

// Inline function to compute cosine using a 9-term Taylor series with SIMD instructions.
// Approximates cos(x) ≈ 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12! - x^14/14! + x^16/16!.
inline __m256d taylorCos_SIMD_9(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                     // Coefficient for x^0 term: 1
    __m256d coeff1 = _mm256_set1_pd(-0.5);                      // Coefficient for x^2 term: -1/2
    __m256d coeff2 = _mm256_set1_pd(0.041666666666666666);      // Coefficient for x^4 term: 1/24
    __m256d coeff3 = _mm256_set1_pd(-0.001388888888888889);     // Coefficient for x^6 term: -1/720
    __m256d coeff4 = _mm256_set1_pd(2.48015873015873e-05);      // Coefficient for x^8 term: 1/40320
    __m256d coeff5 = _mm256_set1_pd(-2.755731922398589e-07);     // Coefficient for x^10 term: -1/3628800
    __m256d coeff6 = _mm256_set1_pd(2.08767569878681e-09);      // Coefficient for x^12 term: 1/479001600
    __m256d coeff7 = _mm256_set1_pd(-1.1470745597729725e-11);   // Coefficient for x^14 term: -1/87178291200
    __m256d coeff8 = _mm256_set1_pd(4.779477332387385e-13);     // Coefficient for x^16 term: 1/20922789888000

    __m256d sum = _mm256_setzero_pd();        // Initialize sum to zero
    __m256d x_power = _mm256_set1_pd(1.0);    // Start with x^0 = 1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);  // Add x^0 term
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);     // Compute x^2 for successive powers

    // Compute x^2 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^2
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);  // Add x^2 term

    // Compute x^4 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^4
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);  // Add x^4 term

    // Compute x^6 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^6
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);  // Add x^6 term

    // Compute x^8 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^8
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);  // Add x^8 term

    // Compute x^10 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^10
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);  // Add x^10 term

    // Compute x^12 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^12
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);  // Add x^12 term

    // Compute x^14 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^14
    sum = _mm256_fmadd_pd(coeff7, x_power, sum);  // Add x^14 term

    // Compute x^16 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^16
    sum = _mm256_fmadd_pd(coeff8, x_power, sum);  // Add x^16 term

    return sum;  // Return the 9-term cosine approximation
}

// Inline function to compute tangent using a 9-term Maclaurin series with SIMD instructions.
// Approximates tan(x) ≈ x + x^3/3 + 2*x^5/15 + 17*x^7/315 + 62*x^9/2835 + 1382*x^11/155925 +
// 21844*x^13/6081075 + 929569*x^15/638512875 + 6404582*x^17/10854718875.
inline __m256d taylorTan_SIMD_9(__m256d x_vec) {
    __m256d coeff0 = _mm256_set1_pd(1.0);                    // Coefficient for x^1 term: 1
    __m256d coeff1 = _mm256_set1_pd(0.3333333333333333);      // Coefficient for x^3 term: 1/3
    __m256d coeff2 = _mm256_set1_pd(0.13333333333333333);     // Coefficient for x^5 term: 2/15
    __m256d coeff3 = _mm256_set1_pd(0.05396825396825397);     // Coefficient for x^7 term: 17/315
    __m256d coeff4 = _mm256_set1_pd(0.021869488536155202);    // Coefficient for x^9 term: 62/2835
    __m256d coeff5 = _mm256_set1_pd(0.0088632355299022);      // Coefficient for x^11 term: 1382/155925
    __m256d coeff6 = _mm256_set1_pd(0.003592128036572);       // Coefficient for x^13 term: 21844/6081075
    __m256d coeff7 = _mm256_set1_pd(0.001455834);             // Coefficient for x^15 term: 929569/638512875
    __m256d coeff8 = _mm256_set1_pd(0.000589041);             // Coefficient for x^17 term: 6404582/10854718875

    __m256d sum = _mm256_setzero_pd();  // Initialize sum to zero
    __m256d x_power = x_vec;            // Start with x^1
    sum = _mm256_fmadd_pd(coeff0, x_power, sum);  // Add x^1 term
    __m256d x2 = _mm256_mul_pd(x_vec, x_vec);     // Compute x^2 for successive powers

    // Compute x^3 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^3
    sum = _mm256_fmadd_pd(coeff1, x_power, sum);  // Add x^3 term

    // Compute x^5 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^5
    sum = _mm256_fmadd_pd(coeff2, x_power, sum);  // Add x^5 term

    // Compute x^7 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^7
    sum = _mm256_fmadd_pd(coeff3, x_power, sum);  // Add x^7 term

    // Compute x^9 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^9
    sum = _mm256_fmadd_pd(coeff4, x_power, sum);  // Add x^9 term

    // Compute x^11 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^11
    sum = _mm256_fmadd_pd(coeff5, x_power, sum);  // Add x^11 term

    // Compute x^13 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^13
    sum = _mm256_fmadd_pd(coeff6, x_power, sum);  // Add x^13 term

    // Compute x^15 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^15
    sum = _mm256_fmadd_pd(coeff7, x_power, sum);  // Add x^15 term

    // Compute x^17 term
    x_power = _mm256_mul_pd(x_power, x2);  // Now x^17
    sum = _mm256_fmadd_pd(coeff8, x_power, sum);  // Add x^17 term

    return sum;  // Return the 9-term tangent approximation
}

// Function to compute sine using a 5-term Taylor series with angle reduction.
// Takes a scalar angle in radians and returns its sine value.
double taylorSin5(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_sin_SIMD(ang_vec);  // Reduce angle using SIMD sine reduction
    __m256d sin_vec = taylorSin_SIMD_5(reduced.red);  // Compute 5-term Taylor sine approximation
    sin_vec = _mm256_mul_pd(sin_vec, reduced.sign);   // Apply sign adjustment from angle reduction
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, sin_vec);  // Extract result (only first element used)
    return result;  // Return the computed sine value
}

// Function to compute cosine using a 5-term Taylor series with angle reduction.
// Takes a scalar angle in radians and returns its cosine value.
double taylorCos5(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_cos_SIMD(ang_vec);  // Reduce angle using SIMD cosine reduction
    __m256d cos_vec = taylorCos_SIMD_5(reduced.red);  // Compute 5-term Taylor cosine approximation
    cos_vec = _mm256_mul_pd(cos_vec, reduced.sign);   // Apply sign adjustment from angle reduction
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, cos_vec);  // Extract result (only first element used)
    return result;  // Return the computed cosine value
}

// Function to compute tangent using a 5-term Maclaurin series with angle reduction.
// Takes a scalar angle in radians and returns its tangent value.
double maclaurinTan5(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_tan_SIMD(ang_vec);  // Reduce angle using SIMD tangent reduction
    __m256d tan_vec = taylorTan_SIMD_5(reduced.red);  // Compute 5-term Maclaurin tangent approximation
    tan_vec = _mm256_mul_pd(tan_vec, reduced.sign);   // Apply sign adjustment from angle reduction
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, tan_vec);  // Extract result (only first element used)
    return result;  // Return the computed tangent value
}

// Function to compute sine using a 7-term Taylor series with angle reduction.
// Takes a scalar angle in radians and returns its sine value.
double taylorSin7(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_sin_SIMD(ang_vec);  // Reduce angle using SIMD sine reduction
    __m256d sin_vec = taylorSin_SIMD_7(reduced.red);  // Compute 7-term Taylor sine approximation
    sin_vec = _mm256_mul_pd(sin_vec, reduced.sign);   // Apply sign adjustment from angle reduction
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, sin_vec);  // Extract result (only first element used)
    return result;  // Return the computed sine value
}

// Function to compute cosine using a 7-term Taylor series with angle reduction.
// Takes a scalar angle in radians and returns its cosine value.
double taylorCos7(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_cos_SIMD(ang_vec);  // Reduce angle using SIMD cosine reduction
    __m256d cos_vec = taylorCos_SIMD_7(reduced.red);  // Compute 7-term Taylor cosine approximation
    cos_vec = _mm256_mul_pd(cos_vec, reduced.sign);   // Apply sign adjustment from angle reduction
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, cos_vec);  // Extract result (only first element used)
    return result;  // Return the computed cosine value
}

// Function to compute tangent using a 7-term Maclaurin series with angle reduction.
// Takes a scalar angle in radians and returns its tangent value.
double maclaurinTan7(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_tan_SIMD(ang_vec);  // Reduce angle using SIMD tangent reduction
    __m256d tan_vec = taylorTan_SIMD_7(reduced.red);  // Compute 7-term Maclaurin tangent approximation
    tan_vec = _mm256_mul_pd(tan_vec, reduced.sign);   // Apply sign adjustment from angle reduction
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, tan_vec);  // Extract result (only first element used)
    return result;  // Return the computed tangent value
}

// Function to compute sine using a 9-term Taylor series with angle reduction.
// Takes a scalar angle in radians and returns its sine value.
double taylorSin9(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_sin_SIMD(ang_vec);  // Reduce angle using SIMD sine reduction
    __m256d sin_vec = taylorSin_SIMD_9(reduced.red);  // Compute 9-term Taylor sine approximation
    sin_vec = _mm256_mul_pd(sin_vec, reduced.sign);   // Apply sign adjustment from angle reduction
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, sin_vec);  // Extract result (only first element used)
    return result;  // Return the computed sine value
}

// Function to compute cosine using a 9-term Taylor series with angle reduction.
// Takes a scalar angle in radians and returns its cosine value.
double taylorCos9(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_cos_SIMD(ang_vec);  // Reduce angle using SIMD cosine reduction
    __m256d cos_vec = taylorCos_SIMD_9(reduced.red);  // Compute 9-term Taylor cosine approximation
    cos_vec = _mm256_mul_pd(cos_vec, reduced.sign);   // Apply sign adjustment from angle reduction
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, cos_vec);  // Extract result (only first element used)
    return result;  // Return the computed cosine value
}

// Function to compute tangent using a 9-term Maclaurin series with angle reduction.
// Takes a scalar angle in radians and returns its tangent value.
double maclaurinTan9(double x) {
    __m256d ang_vec = _mm256_set1_pd(x);  // Broadcast scalar angle to all four lanes of a 256-bit vector
    Vec2 reduced = reduce_angle_tan_SIMD(ang_vec);  // Reduce angle using SIMD tangent reduction
    __m256d tan_vec = taylorTan_SIMD_9(reduced.red);  // Compute 9-term Maclaurin tangent approximation
    tan_vec = _mm256_mul_pd(tan_vec, reduced.sign);   // Apply sign adjustment from angle reduction
    double result;  // Scalar variable to hold the final result
    _mm256_storeu_pd(&result, tan_vec);  // Extract result (only first element used)
    return result;  // Return the computed tangent value
}