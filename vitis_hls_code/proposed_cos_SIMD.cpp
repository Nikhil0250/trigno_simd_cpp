// my_cos.cpp
#include <math.h>
#include "hls_math.h"

#define M_PID 3.14159265358979323846

struct Vec2 {
    double red;
    double sign;
};

// Angle reduction for cos(x)
// (Logic: take x mod 2pi, then adjust so that the reduced angle is in [0, pi/2]
//  and set sign = –1 when x is in the “negative” half.)
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

// Helper for cos: computes (c*ang + d) * fastInverseSqrt(x*ang^2 + y*ang + z)
static inline double cos_helper(double c, double d, double x, double y, double z, double ang) {
    double poly = c * ang + d;
    double ang2 = ang * ang;
    double inner = x * ang2 + y * ang + z;
    double invSqrt = fastInverseSqrt(inner);
    return poly * invSqrt;
}

// Dummy parameter structure and table for cos approximation.
struct CosHelperParams {
    double c;
    double d;
    double x;
    double y;
    double z;
};

static const CosHelperParams cosparams[7] = {
    {-57.29577951, 632.0, 400502.374645736, -72421.86530064, 399424.0},  // Parameters for range [0, intervalWidth)
    {-229.1831181, 657.0, 380805.536587892, -289687.4612814 , 431749.0},  // Parameters for range [intervalWidth, 2*intervalWidth)
    {-286.4788976, 541.0, 200251.187385321, -278342.89691 ,294797.0},  // Parameters for range [2*intervalWidth, 3*intervalWidth)
    {-744.8451337, 1252.0, 883074.908162424, -1616428.5317114 , 1614593.0},  // Parameters for range [3*intervalWidth, 4*intervalWidth)
    {-572.9577951, 910.0, 380805.536587892,-906648.4149306 , 916309.0},  // Parameters for range [4*intervalWidth, 5*intervalWidth)
    {-630.2535746, 990.0, 400502.374645736, -1185793.45271916, 1273864.0} ,  // Parameters for range [5*intervalWidth, RANGE_MAX] (or above)
    {-630.2535746, 990.0, 400502.374645736, -1185793.45271916, 1273864.0}   // Out of Bounds Handling
};

extern "C" {
  void trig_approx(double in, double* out) {
  #pragma HLS INTERFACE s_axilite port=in   bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=out  bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

      Vec2 reduced = reduce_angle_cos(in);
      int idx = (int)(reduced.red / 0.261);
      CosHelperParams p = cosparams[idx];
      double result = cos_helper(p.c, p.d, p.x, p.y, p.z, reduced.red);
      result = result * reduced.sign;
      *out = result;
  }
}
