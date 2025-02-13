// taylor_tan.cpp
#include <math.h>
#include "hls_math.h"

#define M_PID 3.14159265358979323846

struct Vec2 {
    double red;
    double sign;
};

// Angle reduction for tan(x)
static inline Vec2 reduce_angle_tan(double ang) {
    Vec2 ret;
    double pi = M_PID;
    double quotient = floor(ang / pi);
    double ang_mod = ang - quotient * pi;
    if (ang_mod * 2.0 >= pi)
        ang_mod = ang_mod - pi;
    double sign = (ang_mod < 0.0) ? -1.0 : 1.0;
    ang_mod = fabs(ang_mod);
    ret.red = ang_mod;
    ret.sign = sign;
    return ret;
}

// Taylor series for tan(x) using 5 terms:
// tan(x) ≈ x + (1/3)x^3 + (2/15)x^5 + (17/315)x^7 + (62/2835)x^9
double taylor_tan_5terms(double x) {
    double x2 = x * x;
    double x3 = x2 * x;
    double x5 = x3 * x2;
    double x7 = x5 * x2;
    double x9 = x7 * x2;
    return x + (x3 / 3.0) + (2.0 * x5 / 15.0) + (17.0 * x7 / 315.0) + (62.0 * x9 / 2835.0);
}

extern "C" {
  void trig_approx(double in, double* out) {
  #pragma HLS INTERFACE s_axilite port=in   bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=out  bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

      Vec2 reduced = reduce_angle_tan(in);
      // Avoid near π/2 problems
      if (fabs(reduced.red - M_PID/2.0) < 1e-3) {
          *out = 1.0/0.0;  // return infinity
          return;
      }
      double result = taylor_tan_5terms(reduced.red) * reduced.sign;
      *out = result;
  }
}
