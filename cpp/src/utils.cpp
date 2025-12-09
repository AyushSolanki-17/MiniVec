// #pragma once
// #include <cmath>
// #include <cstddef>

// // portable fast squared-L2 (no sqrt)
// static inline float l2_squared_c(const float* __restrict a,
//                                  const float* __restrict b,
//                                  int dim) {
//     double sum = 0.0;
//     int i = 0;
//     int n4 = dim / 4;
//     for (i = 0; i < n4 * 4; i += 4) {
//         double d0 = (double)a[i]     - (double)b[i];
//         double d1 = (double)a[i + 1] - (double)b[i + 1];
//         double d2 = (double)a[i + 2] - (double)b[i + 2];
//         double d3 = (double)a[i + 3] - (double)b[i + 3];
//         sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
//     }
//     for (; i < dim; ++i) {
//         double d = (double)a[i] - (double)b[i];
//         sum += d*d;
//     }
//     return (float)sum;
// }

// // convenience: l2 (sqrt) only if you truly need it
// static inline float l2_c(const float* a, const float* b, int dim) {
//     return std::sqrt(l2_squared_c(a,b,dim));
// }



