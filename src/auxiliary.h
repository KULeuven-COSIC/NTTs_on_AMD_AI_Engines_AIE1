#pragma once

#include "adf/new_frontend/adf.h"
#include "kernels/kernels.h"
#include <utility>


/* Find LOG2, using GenAI */
template <unsigned N, unsigned Count = 0>
struct LOG2 {
    static_assert(N % 2 == 0, "LENGTH must be a power of two");
    static constexpr unsigned value = (N == 1) ? Count : LOG2<N / 2, Count + 1>::value;
};

// Specialization for base case (N == 1)
template <unsigned Count>
struct LOG2<1, Count> {
    static constexpr unsigned value = Count;
};




/* Find number of layers to perform, i.e. the lowest L such that OMEGA^(2^L) mod MODULUS_ = 1,using GenAI */
// Compile-time power function using recursion
template <int BASE, int EXP, int MOD>
struct Power {
    static constexpr int value = (EXP == 0) ? 1 
        : ((EXP % 2 == 0) 
            ? Power<1LL * BASE * BASE % MOD, EXP / 2, MOD>::value
            : 1LL * BASE % MOD);
};

// Template metaprogramming to find the lowest number of layers
template <int MODULUS, int ROOT, int MAX_L = 100, int L = 0, bool Done = (Power<ROOT, (1 << L), MODULUS>::value == 1)>
struct DETERMINE_LOG_ROOT_SIZE {
    static_assert(L <= MAX_L, "ROOT is not correct");
    static constexpr int value = (Done ? L : DETERMINE_LOG_ROOT_SIZE<MODULUS, ROOT, MAX_L, L + 1>::value);
};

// Base case: When Power<OMEGA, 2^L, MODULUS> == 1
template <int MODULUS, int ROOT, int MAX_L, int L>
struct DETERMINE_LOG_ROOT_SIZE<MODULUS, ROOT, MAX_L, L, true> {
    static_assert(L <= MAX_L, "ROOT is not correct");
    static constexpr int value = L;
};


// Find root size
template <int MODULUS, int ROOT>
struct DETERMINE_ROOT_SIZE {
    // take 2-exponent of log root size
    static constexpr int value = 1<<DETERMINE_LOG_ROOT_SIZE<MODULUS, ROOT>::value;
};









/* Find Modular inverse with XGCD, using GenAI */
// Extended GCD - Finds X such that (N * X) ≡ 1 (mod M)
template <int N, int M, int X0 = 1, int X1 = 0, int A = N, int B = M, bool Done = (B == 0)>
struct ExtendedGCD {
    static constexpr int quotient = A / B;
    static constexpr int remainder = A % B;
    static constexpr int nextX = X0 - quotient * X1;

    static constexpr int value = (Done ? (X0 % M + M) % M : ExtendedGCD<N, M, X1, nextX, B, remainder>::value);
};

// Base case: When remainder becomes 0
template <int N, int M, int X0, int X1, int A, int B>
struct ExtendedGCD<N, M, X0, X1, A, B, true> {
    static constexpr int value = (X0 % M + M) % M;
};


/* Construct one kernel per layer for kernel classes that take the layer index LAYER as a template
 * argument (used for the inverse radix-2 int16 pipeline). The index must be a compile-time constant,
 * hence the std::index_sequence fold expression. */
template <int i, template<unsigned, unsigned, unsigned, unsigned, unsigned> class KERNEL_CLASS, std::size_t NO_OF_LAYERS, unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
void construct_kernel_regular(adf::kernel* layers, std::vector<std::vector<int16>>& twiddle_factors, std::vector<std::vector<int16>>& barrett_multipliers) {
    layers[i] = adf::kernel::create_object<KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE, NO_OF_LAYERS, i>>(twiddle_factors[i], barrett_multipliers[i], std::vector<int16>(), std::vector<int16>());
}
template <template<unsigned, unsigned, unsigned, unsigned, unsigned> class KERNEL_CLASS, std::size_t NO_OF_LAYERS, unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE, size_t... Is>
void construct_kernel_for_indices_regular(std::index_sequence<Is...>, adf::kernel* layers, std::vector<std::vector<int16>>& twiddle_factors, std::vector<std::vector<int16>>& barrett_multipliers) {
    (construct_kernel_regular<Is, KERNEL_CLASS, NO_OF_LAYERS, LENGTH, MODULUS, VECTOR_SIZE>(layers, twiddle_factors, barrett_multipliers), ...);  // Fold expression to call execute<Is>() for each Is
}
template <template<unsigned, unsigned, unsigned, unsigned, unsigned> class KERNEL_CLASS, std::size_t NO_OF_LAYERS, unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
void construct_layer_kernels_regular(adf::kernel* layers, std::vector<std::vector<int16>>& twiddle_factors, std::vector<std::vector<int16>>& barrett_multipliers) {
    construct_kernel_for_indices_regular<KERNEL_CLASS, NO_OF_LAYERS, LENGTH, MODULUS, VECTOR_SIZE>(std::make_index_sequence<NO_OF_LAYERS>{}, layers, twiddle_factors, barrett_multipliers);
}
