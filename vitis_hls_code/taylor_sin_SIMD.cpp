// taylor_sin.cpp
#include <math.h>
#include "hls_math.h"

#define M_PID 3.14159265358979323846

// Structure for angle reduction result.
struct Vec2 {
    double red;
    double sign;
};

// Angle reduction for sin(x)
static inline Vec2 reduce_angle_sin(double ang) {
    Vec2 ret;
    double two_pi = 2.0 * M_PID;
    double quotient = floor(ang / two_pi);
    double ang_mod = ang - quotient * two_pi;
    double sign = (ang_mod < 0.0) ? -1.0 : 1.0;
    double x = fabs(ang_mod);
    if (x > M_PID) {
        sign = -sign;
        x = x - M_PID;
    }
    if (x * 2.0 >= M_PID) {
        x = M_PID - x;
    }
    ret.red = x;
    ret.sign = sign;
    return ret;
}

// Taylor (Maclaurin) series for sin(x) using 5 terms:
// sin(x) ≈ x - x^3/6 + x^5/120 - x^7/5040 + x^9/362880
double taylor_sin_5terms(double x) {
    double x2 = x * x;
    double x3 = x2 * x;
    double x5 = x3 * x2;
    double x7 = x5 * x2;
    double x9 = x7 * x2;
    return x - (x3 / 6.0) + (x5 / 120.0) - (x7 / 5040.0) + (x9 / 362880.0);
}

extern "C" {
  void trig_approx(double in, double* out) {
  #pragma HLS INTERFACE s_axilite port=in   bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=out  bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

      Vec2 reduced = reduce_angle_sin(in);
      double result = taylor_sin_5terms(reduced.red) * reduced.sign;
      *out = result;
  }
}
