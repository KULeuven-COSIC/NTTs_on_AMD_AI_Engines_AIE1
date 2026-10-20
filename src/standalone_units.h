
#pragma once

#include <adf.h>
#include "kernels/kernels.h"
#include "kernels/debug.h"
#include "auxiliary.h"

using namespace adf;

/*
 * Single-tile graphs for the stand-alone units, so that they can be benchmarked with the same
 * input/output harness (DUT_GRAPH in project.cpp) as the NTT pipelines.
 */


/* Bit-reversal unit for int16 (Section 6, Table 6). LENGTH must be 256, 512, 1024, 2048 or 4096. */
template<unsigned LENGTH>
class BITREVERSE : public adf::graph {
private:
    kernel bitrev;

public:
    input_port in_data;
    output_port out_data;

    BITREVERSE() {
        bitrev = kernel::create(bitreverse_improved_kernel<LENGTH>);
        source(bitrev) = "kernels/bitreverse_kernels.cc";
        runtime<ratio>(bitrev) = 1.0;
        dimensions(bitrev.in[0]) = { LENGTH };
        dimensions(bitrev.out[0]) = { LENGTH };

        connect(in_data, bitrev.in[0]);
        connect(bitrev.out[0], out_data);
    }
};


/* Full Barrett multiplication by a constant for int16 (Section 4.2, Table 1): out = FACTOR * in mod q in [0, q). */
template<unsigned LENGTH, unsigned MODULUS, unsigned FACTOR>
class FULL_BARRETT_MULTIPLICATION : public adf::graph {
private:
    kernel full_barrett;

public:
    input_port in_data;
    output_port out_data;

    FULL_BARRETT_MULTIPLICATION() {
        full_barrett = kernel::create_object<FULL_BARRETT_MULTIPLICATION_KERNEL_CLASS<LENGTH, MODULUS, FACTOR>>(std::vector<int16>(), std::vector<int16>());
        source(full_barrett) = "kernels/full_reduction_kernels.cc";
        initialization_function(full_barrett) = "set_modes_final_reduction";
        runtime<ratio>(full_barrett) = 1.0;
        dimensions(full_barrett.in[0]) = { LENGTH };
        dimensions(full_barrett.out[0]) = { LENGTH };

        connect(in_data, full_barrett.in[0]);
        connect(full_barrett.out[0], out_data);
    }
};
