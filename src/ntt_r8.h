#pragma once

#include <adf.h>
#include "kernels/kernels.h"
#include "kernels/debug.h"
#include "auxiliary.h"
#include "twiddle.h"

using namespace adf;

/*
 * Radix-8 int16 NTT, one pipeline stage (Section 5.3 of the paper).
 *
 * One radix-8 layer is split over two tiles because its buffers do not fit in four memory banks:
 *   layers1    : radix-8 butterflies without reduction (int16 -> int32)
 *   reduction1 : 32-to-16-bit Barrett multiplication with the between-layer twiddle factors (int32 -> int16)
 * This graph instantiates one such stage (2 tiles). A full transform chains 2*ceil(log8(n)) tiles;
 * since every stage runs the same kernels, the throughput of the pipeline equals the throughput of
 * the slower of these two tiles, which is what this graph measures.
 */

template<unsigned LENGTH, unsigned MODULUS, unsigned ROOT>
class NTT_R8 : public adf::graph {

private:
  kernel layers1;
  kernel reduction1;

public:
 input_port in_data;
 output_port out_data;



    NTT_R8() {

        // Determine the factors
        std::vector<std::vector<int16>> twiddle_factors_layers(1);
        std::vector<std::vector<int16>> twiddle_factors_reduction(1);
        std::vector<std::vector<int32>> barrett_factors_reduction(1);
        determine_factors_r8<MODULUS, ROOT, LENGTH>(twiddle_factors_layers, twiddle_factors_reduction, barrett_factors_reduction);

        layers1 = kernel::create_object<NTT_R8_LAYERS_KERNEL_CLASS<MODULUS, LENGTH, 16>>(twiddle_factors_layers[0], std::vector<int32>());
        source(layers1) = "kernels/ntt_r8_kernels.cc";
        runtime<ratio>(layers1) = 1.0;
        dimensions(layers1.in[0]) = { LENGTH };
        dimensions(layers1.out[0]) = { LENGTH };

        reduction1 = kernel::create_object<NTT_R8_RECUTION_KERNEL_CLASS<MODULUS, LENGTH, 16>>(barrett_factors_reduction[0], twiddle_factors_reduction[0], std::vector<int16>(), std::vector<int32>());
        source(reduction1) = "kernels/ntt_r8_kernels.cc";
        runtime<ratio>(reduction1) = 1.0;
        dimensions(reduction1.in[0]) = { LENGTH };
        dimensions(reduction1.out[0]) = { LENGTH };

        connect(in_data, layers1.in[0]);
        connect(layers1.out[0], reduction1.in[0]);
        connect(reduction1.out[0], out_data);
    }


};
