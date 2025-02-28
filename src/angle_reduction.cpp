// Include the corresponding header file for angle reduction function declarations.
// Ensures consistency between function definitions and their declarations.
#include "angle_reduction.h"

// Inline function to compute the absolute value of a 256-bit vector of doubles.
// Uses a branchless method with a sign mask to clear the sign bit of each element.
inline __m256d abs256(__m256d x) {
    __m256d mask = _mm256_set1_pd(-0.0);  // Sign mask: -0.0 has all bits set in the sign position
    return _mm256_andnot_pd(mask, x);     // Clear sign bit by AND NOT operation, preserving magnitude
}

// Function to reduce angles for sine computation using SIMD instructions.
// Reduces angles to the range [0, π/2] and adjusts signs based on the original quadrant.
Vec2 reduce_angle_sin_SIMD(__m256d ang) {
    Vec2 ret;                             // Structure to hold reduced angle and sign
    __m256d two_pi = _mm256_set1_pd(2 * M_PID);  // Constant vector with 2π for periodicity
    __m256d quotient = _mm256_div_pd(ang, two_pi);  // Compute number of 2π periods
    quotient = _mm256_round_pd(quotient, _MM_FROUND_FLOOR);  // Round down to integer periods
    __m256d ang_mod = _mm256_sub_pd(ang, _mm256_mul_pd(quotient, two_pi));  // Subtract periods

    // Determine initial sign based on whether reduced angle is negative
    __m256d sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0),
                                    _mm256_cmp_pd(ang_mod, _mm256_set1_pd(0.0), _CMP_LT_OQ));
    __m256d x = abs256(ang_mod);  // Take absolute value of the reduced angle

    // If angle > π, adjust sign and subtract π to map to [0, π]
    __m256d mask_gt_pi = _mm256_cmp_pd(x, _mm256_set1_pd(M_PID), _CMP_GT_OQ);
    sign = _mm256_mul_pd(sign,
             _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0), mask_gt_pi));
    x = _mm256_sub_pd(x, _mm256_and_pd(mask_gt_pi, _mm256_set1_pd(M_PID)));

    // If angle ≥ π/2, reflect to [0, π/2] using π - x
    __m256d mask_ge_half_pi = _mm256_cmp_pd(_mm256_mul_pd(x, _mm256_set1_pd(2.0)),
                                             _mm256_set1_pd(M_PID), _CMP_GE_OQ);
    x = _mm256_blendv_pd(x, _mm256_sub_pd(_mm256_set1_pd(M_PID), x), mask_ge_half_pi);

    ret.red  = x;     // Store final reduced angle
    ret.sign = sign;  // Store final sign adjustment
    return ret;       // Return reduced angle and sign
}

// Function to reduce angles for cosine computation using SIMD instructions.
// Reduces angles to the range [0, π/2] and adjusts signs based on the original quadrant.
Vec2 reduce_angle_cos_SIMD(__m256d ang) {
    Vec2 ret;                             // Structure to hold reduced angle and sign
    __m256d two_pi = _mm256_set1_pd(2 * M_PID);  // Constant vector with 2π for periodicity
    __m256d quotient = _mm256_div_pd(ang, two_pi);  // Compute number of 2π periods
    quotient = _mm256_round_pd(quotient, _MM_FROUND_FLOOR);  // Round down to integer periods
    __m256d ang_mod = _mm256_sub_pd(ang, _mm256_mul_pd(quotient, two_pi));  // Subtract periods

    __m256d pi_half = _mm256_set1_pd(M_PID / 2.0);       // Constant vector with π/2
    __m256d three_pi_half = _mm256_set1_pd(3 * M_PID / 2.0);  // Constant vector with 3π/2

    // Determine sign: negative if angle is in (π/2, 3π/2]
    __m256d cmp1 = _mm256_cmp_pd(ang_mod, pi_half, _CMP_GT_OQ);
    __m256d cmp2 = _mm256_cmp_pd(ang_mod, three_pi_half, _CMP_LE_OQ);
    __m256d mask = _mm256_and_pd(cmp1, cmp2);
    __m256d sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0), mask);

    // If angle > π, map to [0, π] using 2π - x
    __m256d mask_gt_pi = _mm256_cmp_pd(ang_mod, _mm256_set1_pd(M_PID), _CMP_GT_OQ);
    ang_mod = _mm256_blendv_pd(ang_mod, _mm256_sub_pd(two_pi, ang_mod), mask_gt_pi);
    
    // If angle > π/2, map to [0, π/2] using π - x
    __m256d mask_gt_pi_half = _mm256_cmp_pd(ang_mod, pi_half, _CMP_GT_OQ);
    ang_mod = _mm256_blendv_pd(ang_mod, _mm256_sub_pd(_mm256_set1_pd(M_PID), ang_mod), mask_gt_pi_half);

    ret.red  = ang_mod;  // Store final reduced angle
    ret.sign = sign;     // Store final sign adjustment
    return ret;          // Return reduced angle and sign
}

// Function to reduce angles for tangent computation using SIMD instructions.
// Reduces angles to the range [0, π/2) and adjusts signs based on the original quadrant.
Vec2 reduce_angle_tan_SIMD(__m256d ang) {
    Vec2 ret;                             // Structure to hold reduced angle and sign
    __m256d pi = _mm256_set1_pd(M_PID);   // Constant vector with π for periodicity
    __m256d quotient = _mm256_div_pd(ang, pi);  // Compute number of π periods
    quotient = _mm256_round_pd(quotient, _MM_FROUND_FLOOR);  // Round down to integer periods
    __m256d ang_mod = _mm256_sub_pd(ang, _mm256_mul_pd(quotient, pi));  // Subtract periods

    // If angle ≥ π/2, subtract π to map to [0, π/2)
    __m256d mask = _mm256_cmp_pd(_mm256_mul_pd(ang_mod, _mm256_set1_pd(2.0)),
                                  pi, _CMP_GE_OQ);
    ang_mod = _mm256_blendv_pd(ang_mod, _mm256_sub_pd(ang_mod, pi), mask);

    // Determine sign based on whether reduced angle is negative
    __m256d sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0),
                                    _mm256_cmp_pd(ang_mod, _mm256_set1_pd(0.0), _CMP_LT_OQ));
    ang_mod = abs256(ang_mod);  // Take absolute value of the reduced angle

    ret.red  = ang_mod;  // Store final reduced angle
    ret.sign = sign;     // Store final sign adjustment
    return ret;          // Return reduced angle and sign
}