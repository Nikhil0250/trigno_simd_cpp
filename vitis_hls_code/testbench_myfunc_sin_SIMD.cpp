#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Declaration of the top-level function from my_sin.cpp
extern "C" {
    void trig_approx(double in, double* out);
}

int main() {
    const int numAngles   = 100;
    const int iterations  = 10000;
    double angle, output;
    
    // Seed random number generator for reproducibility
    srand(2024873);
    
    for (int i = 0; i < numAngles; i++) {
        // Generate a random angle in the range [-1e6, 1e6]
        angle = ((double)rand() / RAND_MAX) * 2e6 - 1e6;
        for (int j = 0; j < iterations; j++) {
            trig_approx(angle, &output);
        }
        // printf("my_sin: Angle = %f, Output (last iteration) = %f\n", angle, output);
    }
    return 0;
}
