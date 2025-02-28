#ifndef MYFUNCTIONS_H
#define MYFUNCTIONS_H

#include<immintrin.h>
#include"angle_reduction.h"

struct SinHelperParams {
    double a;
    double b;
    double x;
    double y;
    double z;
};

struct CosHelperParams {
    double c;
    double d;
    double x;
    double y;
    double z;
};

struct TanHelperParams {
    double a;
    double b;
    double c;
    double d;
};

extern SinHelperParams sinparams[7];
extern CosHelperParams cosparams[7];
extern TanHelperParams tanparams[7];

inline __m256d fastInverseSqrt_SIMD(__m256d number);
inline __m256d sin_helper_SIMD(__m256d a, __m256d b, __m256d x, __m256d y, __m256d z, __m256d ang);
inline __m256d cos_helper_SIMD(__m256d c, __m256d d,  __m256d x, __m256d y, __m256d z, __m256d ang);
inline __m256d tan_helper_SIMD(__m256d a, __m256d b,__m256d c, __m256d d, __m256d ang);

double proposed_sin(double ang);
double proposed_cos(double ang);
double proposed_tan(double ang);

#endif  // MYFUNCTIONS_H
