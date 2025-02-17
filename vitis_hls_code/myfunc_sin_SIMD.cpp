// my_sin.cpp
#include <math.h>
#include "hls_math.h"  // Vitis HLS math functions

#define M_PID 3.14159265358979323846

// A simple structure to return (reduced angle, sign)
struct Vec2 {
    double red;
    double sign;
};

// Angle reduction for sin(x)
// (Logic: x mod 2pi, then adjust sign so that final x is in [0, pi/2].)
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

// Fast inverse square-root (scalar version) using the “magic number” method
static inline double fastInverseSqrt(double number) {
    double x2 = number * 0.5;
    double y  = number;
    // use a union to do bit-level reinterpretation
    union { double d; long long ll; } u;
    u.d = y;
    u.ll = 0x5fe6eb50c7b537a9LL - (u.ll >> 1);
    y = u.d;
    y = y * (1.5 - (x2 * y * y));
    return y;
}

// Helper that computes: (a*ang + b) * fastInverseSqrt(x*ang^2 + y*ang + z)
static inline double sin_helper(double a, double b, double x, double y, double z, double ang) {
    double poly = a * ang + b;
    double ang2 = ang * ang;
    double inner = x * ang2 + y * ang + z;
    double invSqrt = fastInverseSqrt(inner);
    return poly * invSqrt;
}

// Dummy parameter structure and table for sin approximation.
// (In a real design, fill in these parameters with your pre–computed coefficients.)
struct SinHelperParams {
    double a;
    double b;
    double x;
    double y;
    double z;
};

static const SinHelperParams sinparams[7] = {
    {630.2535746, 0.0, 400502.374645736, -72421.86530064, 399424.0},  // Parameters for range [0, intervalWidth)
    {572.9577951, 10.0, 380805.536587892, -289687.4612814 , 431749.0},  // Parameters for range [intervalWidth, 2*intervalWidth)
    {343.7746771, 46.0, 200251.187385321, -278342.89691 ,294797.0},  // Parameters for range [2*intervalWidth, 3*intervalWidth)
    {572.9577951, 217.0, 883074.908162424, -1616428.5317114 , 1614593.0},  // Parameters for range [3*intervalWidth, 4*intervalWidth)
    {229.1831181, 297.0, 380805.536587892,-906648.4149306 , 916309.0},  // Parameters for range [4*intervalWidth, 5*intervalWidth)
    {57.29577951, 542.0, 400502.374645736, -1185793.45271916, 1273864.0},   // Parameters for range [5*intervalWidth, RANGE_MAX] (or above)
    {57.29577951, 542.0, 400502.374645736, -1185793.45271916, 1273864.0}   // Out of bounds Handling
};

extern "C" {
  // Top-level function for my_sin
  // (AXI–lite interface so that Vitis HLS can report resource usage.)
  void trig_approx(double in, double* out) {
  #pragma HLS INTERFACE s_axilite port=in   bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=out  bundle=CTRL
  #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

      Vec2 reduced = reduce_angle_sin(in);
      // Determine the parameter table index (each interval is ~0.261 rad)
      int idx = (int)(reduced.red / 0.261);
      SinHelperParams p = sinparams[idx];
      double result = sin_helper(p.a, p.b, p.x, p.y, p.z, reduced.red);
      result = result * reduced.sign;
      *out = result;
  }
}
