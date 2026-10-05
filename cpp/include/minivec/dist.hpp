/**
 * @file distance.hpp
 * @brief Declaration of distance functions to find distance between two vectors
 *
 * @details
 * This file defines the public distance functions to find distance between two or more vectors.
 * Part of the MiniVec project.
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace minivec
{
    enum class DistanceMetric
    {
        L2,
        L2Squared,
        Cosine,
        InnerProduct
    };

    // Computes the squared L2 (Euclidean) distance between two float vectors.
    //
    // Uses a loop-unrolled float implementation that compilers can
    // auto-vectorize on the build target.
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
        float sum = 0.0f;
        int i = 0;
        const int unroll = 4;
        // Unroll by 4 to simulate faster calculation.
        int limit = dim - (dim % unroll);

        // Main loop: process 4 elements per iteration to help auto-vectorization.
        for (; i < limit; i += unroll)
        {
            float d0 = a[i] - b[i];
            float d1 = a[i + 1] - b[i + 1];
            float d2 = a[i + 2] - b[i + 2];
            float d3 = a[i + 3] - b[i + 3];
            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        // Remainder loop for dimensions not divisible by 4.
        for (; i < dim; ++i)
        {
            float d = a[i] - b[i];
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

    inline float inner_product_distance(const float *a, const float *b, int dim)
    {
        float dot = 0.0f;
        for (int i = 0; i < dim; ++i)
            dot += a[i] * b[i];
        return -dot;
    }

    inline float cosine_distance(const float *a, const float *b, int dim)
    {
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        for (int i = 0; i < dim; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a == 0.0f || norm_b == 0.0f)
            return 1.0f;
        const float similarity = dot / std::sqrt(norm_a * norm_b);
        return 1.0f - std::max(-1.0f, std::min(1.0f, similarity));
    }

    inline float compute_distance(DistanceMetric metric, const float *a, const float *b, int dim)
    {
        switch (metric)
        {
        case DistanceMetric::L2:
            return l2_distance(a, b, dim);
        case DistanceMetric::L2Squared:
            return l2_squared_distance(a, b, dim);
        case DistanceMetric::Cosine:
            return cosine_distance(a, b, dim);
        case DistanceMetric::InnerProduct:
            return inner_product_distance(a, b, dim);
        }
        throw std::invalid_argument("Unknown distance metric");
    }

    // Function to map string to function pointer
    // NOTE: Add new functions here to supported distance function names only.
    inline DistanceMetric get_distance_metric(const std::string &name)
    {
        if (name == "l2")
            return DistanceMetric::L2;
        if (name == "l2_squared")
            return DistanceMetric::L2Squared;
        if (name == "cosine")
            return DistanceMetric::Cosine;
        if (name == "inner_product")
            return DistanceMetric::InnerProduct;
        throw std::invalid_argument("Unsupported distance function: " + name);
    }

}
