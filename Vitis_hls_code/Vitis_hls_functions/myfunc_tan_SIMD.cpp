// my_tan.cpp
#include <math.h>
#include "hls_math.h"
#include <stdlib.h>

#define M_PID 3.14159265358979323846

struct Vec2 {
    double red;
    double sign;
};

// Angle reduction for tan(x)
// (Reduce x mod π, then adjust so that x is in [0, π/2] and set the sign.)
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

static inline double fastInverseSqrt(double number) {
    double x2 = number * 0.5;
    double y  = number;
    union { double d; long long ll; } u;
    u.d = y;
    u.ll = 0x5fe6eb50c7b537a9LL - (u.ll >> 1);
    y = u.d;
    y = y * (1.5 - (x2 * y * y));
    return y;
}

// Helper for tan: computes ((a*ang)+b) * [fastInverseSqrt(c*ang+d)]^2
static inline double tan_helper(double a, double b, double c, double d, double ang) {
    double poly = a * ang + b;
    double tmp = fastInverseSqrt(c * ang + d);
    return poly * tmp * tmp;
}

// Dummy parameter structure and table for tan approximation.
struct TanHelperParams {
    double a;
    double b;
    double c;
    double d;
};

static const TanHelperParams tanparams[7] = {
    {630.2535746, 0.0, -57.29577951, 632.0},  // Parameters for range [0, intervalWidth)
    {572.9577951, 10.0, -229.1831181, 657.0},  // Parameters for range [intervalWidth, 2*intervalWidth)
    {343.7746771, 46.0, -286.4788976, 541.0},  // Parameters for range [2*intervalWidth, 3*intervalWidth)
    {572.9577951, 217.0, -744.8451337, 1252.0},  // Parameters for range [3*intervalWidth, 4*intervalWidth)
    {229.1831181, 297.0, -572.9577951, 910.0},  // Parameters for range [4*intervalWidth, 5*intervalWidth)
    {57.29577951, 542.0, -630.2535746, 990.0},   // Parameters for range [5*intervalWidth, RANGE_MAX] (or above)
    {57.29577951, 542.0, -630.2535746, 990.0} //out of bounds 
};

extern "C" {
  void trig_approx(double in, double* out) {
  #pragma HLS INTERFACE s_axilite port=in   bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=out  bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

      Vec2 reduced = reduce_angle_tan(in);
      // Avoid division-by-zero issues near π/2.
      if (fabs(reduced.red - M_PID / 2.0) < 1e-3) {
          *out = 1.0/0.0; // infinity
          return;
      }
      int idx = (int)(reduced.red / 0.261);
      TanHelperParams p = tanparams[idx];
      double result = tan_helper(p.a, p.b, p.c, p.d, reduced.red);
      result = result * reduced.sign;
      *out = result;
  }
}
