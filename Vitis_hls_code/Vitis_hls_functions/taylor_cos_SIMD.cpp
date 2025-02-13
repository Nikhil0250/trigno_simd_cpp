// taylor_cos.cpp
#include <math.h>
#include "hls_math.h"

#define M_PID 3.14159265358979323846

struct Vec2 {
    double red;
    double sign;
};

// Angle reduction for cos(x)
static inline Vec2 reduce_angle_cos(double ang) {
    Vec2 ret;
    double two_pi = 2.0 * M_PID;
    double quotient = floor(ang / two_pi);
    double ang_mod = ang - quotient * two_pi;
    double pi_half = M_PID / 2.0;
    double three_pi_half = 3.0 * M_PID / 2.0;
    double sign = ((ang_mod > pi_half) && (ang_mod <= three_pi_half)) ? -1.0 : 1.0;
    if (ang_mod > M_PID)
        ang_mod = two_pi - ang_mod;
    if (ang_mod > pi_half)
        ang_mod = M_PID - ang_mod;
    ret.red = ang_mod;
    ret.sign = sign;
    return ret;
}

// Taylor series for cos(x) using 5 terms:
// cos(x) ≈ 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320
double taylor_cos_5terms(double x) {
    double x2 = x * x;
    double x4 = x2 * x2;
    double x6 = x4 * x2;
    double x8 = x6 * x2;
    return 1.0 - (x2 / 2.0) + (x4 / 24.0) - (x6 / 720.0) + (x8 / 40320.0);
}

extern "C" {
  void trig_approx(double in, double* out) {
  #pragma HLS INTERFACE s_axilite port=in   bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=out  bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

      Vec2 reduced = reduce_angle_cos(in);
      double result = taylor_cos_5terms(reduced.red) * reduced.sign;
      *out = result;
  }
}
