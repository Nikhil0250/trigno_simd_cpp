#include "angle_reduction.h"

// Compute absolute value (branchless)
inline __m256d abs256(__m256d x) {
    __m256d mask = _mm256_set1_pd(-0.0);  // sign mask: -0.0 has all bits set in the sign
    return _mm256_andnot_pd(mask, x);
}

// --- For sin(x) ---
Vec2 reduce_angle_sin_SIMD(__m256d ang) {
    Vec2 ret;
    __m256d two_pi = _mm256_set1_pd(2 * M_PID);
    __m256d quotient = _mm256_div_pd(ang, two_pi);
    quotient = _mm256_round_pd(quotient, _MM_FROUND_FLOOR);
    __m256d ang_mod = _mm256_sub_pd(ang, _mm256_mul_pd(quotient, two_pi));

    __m256d sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0),
                                    _mm256_cmp_pd(ang_mod, _mm256_set1_pd(0.0), _CMP_LT_OQ));
    __m256d x = abs256(ang_mod);
    
    __m256d mask_gt_pi = _mm256_cmp_pd(x, _mm256_set1_pd(M_PID), _CMP_GT_OQ);
    sign = _mm256_mul_pd(sign,
             _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0), mask_gt_pi));
    x = _mm256_sub_pd(x, _mm256_and_pd(mask_gt_pi, _mm256_set1_pd(M_PID)));

    __m256d mask_ge_half_pi = _mm256_cmp_pd(_mm256_mul_pd(x, _mm256_set1_pd(2.0)),
                                             _mm256_set1_pd(M_PID), _CMP_GE_OQ);
    x = _mm256_blendv_pd(x, _mm256_sub_pd(_mm256_set1_pd(M_PID), x), mask_ge_half_pi);

    ret.red  = x;
    ret.sign = sign;
    return ret;
}

// --- For cos(x) ---
Vec2 reduce_angle_cos_SIMD(__m256d ang) {
    Vec2 ret;
    __m256d two_pi = _mm256_set1_pd(2 * M_PID);
    __m256d quotient = _mm256_div_pd(ang, two_pi);
    quotient = _mm256_round_pd(quotient, _MM_FROUND_FLOOR);
    __m256d ang_mod = _mm256_sub_pd(ang, _mm256_mul_pd(quotient, two_pi));

    __m256d pi_half = _mm256_set1_pd(M_PID / 2.0);
    __m256d three_pi_half = _mm256_set1_pd(3 * M_PID / 2.0);

    __m256d cmp1 = _mm256_cmp_pd(ang_mod, pi_half, _CMP_GT_OQ);
    __m256d cmp2 = _mm256_cmp_pd(ang_mod, three_pi_half, _CMP_LE_OQ);
    __m256d mask = _mm256_and_pd(cmp1, cmp2);
    __m256d sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0), mask);

    __m256d mask_gt_pi = _mm256_cmp_pd(ang_mod, _mm256_set1_pd(M_PID), _CMP_GT_OQ);
    ang_mod = _mm256_blendv_pd(ang_mod, _mm256_sub_pd(two_pi, ang_mod), mask_gt_pi);
    
    __m256d mask_gt_pi_half = _mm256_cmp_pd(ang_mod, pi_half, _CMP_GT_OQ);
    ang_mod = _mm256_blendv_pd(ang_mod, _mm256_sub_pd(_mm256_set1_pd(M_PID), ang_mod), mask_gt_pi_half);

    ret.red  = ang_mod;
    ret.sign = sign;
    return ret;
}

// --- For tan(x) ---
Vec2 reduce_angle_tan_SIMD(__m256d ang) {
    Vec2 ret;
    __m256d pi = _mm256_set1_pd(M_PID);
    __m256d quotient = _mm256_div_pd(ang, pi);
    quotient = _mm256_round_pd(quotient, _MM_FROUND_FLOOR);
    __m256d ang_mod = _mm256_sub_pd(ang, _mm256_mul_pd(quotient, pi));

    __m256d mask = _mm256_cmp_pd(_mm256_mul_pd(ang_mod, _mm256_set1_pd(2.0)),
                                  pi, _CMP_GE_OQ);
    ang_mod = _mm256_blendv_pd(ang_mod, _mm256_sub_pd(ang_mod, pi), mask);

    __m256d sign = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(-1.0),
                                    _mm256_cmp_pd(ang_mod, _mm256_set1_pd(0.0), _CMP_LT_OQ));
    ang_mod = abs256(ang_mod);

    ret.red  = ang_mod;
    ret.sign = sign;
    return ret;
}
