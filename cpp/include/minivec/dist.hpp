/**
 * @file distance.hpp
 * @brief Declaration of distance functions to find distance between two vectors
 *
 * @details
 * This file defines the public distance functions to find distance between two or more vectors.
 * Part of the MiniVec project.
 */

#pragma once
#include <cmath>
#include <cstddef>

namespace minivec
{

    // Computes the squared L2 (Euclidean) distance between two float vectors.
    //
    // Uses a scalar implementation written to be friendly to compiler
    // auto-vectorization, with loop unrolling by 4. Accumulation is done in
    // double precision to reduce numerical error.
    //
    // Args:
    //   a: Pointer to the first input vector (size at least dim).
    //   b: Pointer to the second input vector (size at least dim).
    //   dim: Number of elements in each vector.
    //
    // Returns:
    //   Squared L2 distance between a and b.
    inline float l2_squared_scalar(const float *__restrict a,
                                   const float *__restrict b,
                                   int dim)
    {
        double sum = 0.0;
        int i = 0;
        const int unroll = 4;
        // Unroll by 4 to simulate faster calculation.
        int limit = dim - (dim % unroll);

        // Main loop: process 4 elements per iteration to help auto-vectorization.
        for (; i < limit; i += unroll)
        {
            double d0 = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            double d1 = static_cast<double>(a[i + 1]) - static_cast<double>(b[i + 1]);
            double d2 = static_cast<double>(a[i + 2]) - static_cast<double>(b[i + 2]);
            double d3 = static_cast<double>(a[i + 3]) - static_cast<double>(b[i + 3]);
            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        // Remainder loop for dimensions not divisible by 4.
        for (; i < dim; ++i)
        {
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += d * d;
        }
        return static_cast<float>(sum);
    }

    // Computes the squared L2 distance between two float vectors.
    //
    // Thin wrapper around l2_squared_scalar for a user-facing name.
    //
    // Args:
    //   a: Pointer to the first input vector (size at least dim).
    //   b: Pointer to the second input vector (size at least dim).
    //   dim: Number of elements in each vector.
    //
    // Returns:
    //   Squared L2 distance between a and b.
    inline float l2_squared_distance(const float *a, const float *b, int dim)
    {
        return l2_squared_scalar(a, b, dim);
    }

    // Computes the L2 (Euclidean) distance between two float vectors.
    //
    // Computes the squared L2 distance and then takes its square root.
    //
    // Args:
    //   a: Pointer to the first input vector (size at least dim).
    //   b: Pointer to the second input vector (size at least dim).
    //   dim: Number of elements in each vector.
    //
    // Returns:
    //   L2 distance between a and b.
    inline float l2_distance(const float *a, const float *b, int dim)
    {
        return std::sqrt(l2_squared_distance(a, b, dim));
    }

}