#ifndef CORDIC_TRIG_H
#define CORDIC_TRIG_H

#include <vector>
#include <cmath>


// Number of CORDIC iterations for precision
const int ITERATIONS = 20;

// Constant for π
const double PI = acos(-1.0);

// Precompute arctangent values
std::vector<double> precompute_atan(int n);

// Compute sine using CORDIC
double cordic_sin(double angle);

// Compute cosine using CORDIC
double cordic_cos(double angle);

// Compute tangent using CORDIC
double cordic_tan(double angle);

#endif  // CORDIC_TRIG_H
