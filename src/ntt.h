
#pragma once

#include <adf.h>
#include "kernels/kernels.h"
#include "kernels/debug.h"
#include "auxiliary.h"
#include "twiddle.h"

using namespace adf;

/*
 * Radix-2 int16 NTT pipeline (Section 5.1 of the paper).
 *
 * The full transform of size LENGTH is computed by a pipeline of log2(LENGTH) AIE tiles, one radix-2
 * layer per tile, placed in one row starting at tile (STARTCOL, STARTROW). Consecutive tiles exchange
 * the polynomial through shared memory buffers (no streaming interconnect). Each tile stores only
 * the twiddle factors and Barrett multipliers of its own layer, computed at graph-construction time.
 *
 * Template parameters:
 *   LENGTH       polynomial size n (power of two, n <= 4096)
 *   MODULUS      prime modulus q, q < 2^16 / 3 (lazy reduction bound, Table 3)
 *   ROOT         primitive LENGTH-th root of unity modulo q
 *   STARTROW/COL location of the first layer in the AIE array
 *   TYPE         forward (CT, natural order -> bit-reversed order) or
 *                inverse (bit-reversed order -> natural order) transform
 */


enum class NTT_TYPE {
    TYPE_NTT_NO_to_BO,      // forward NTT, natural-order input, bit-reversed output (CT, perfect shuffle per layer)
    TYPE_INTT_BO_to_NO,     // inverse NTT, bit-reversed input, natural-order output (unshuffle per layer)
};

// Is it an inverse transform
constexpr bool is_inverse_transform(NTT_TYPE TYPE) {
    return TYPE == NTT_TYPE::TYPE_INTT_BO_to_NO;
}



/* NTT GRAPH CLASS */
template <uint16 LENGTH, uint16 MODULUS, uint16 ROOT, uint16 STARTROW, uint16 STARTCOL, NTT_TYPE TYPE, uint16 NO_OF_LAYERS=DETERMINE_LOG_ROOT_SIZE<MODULUS, ROOT>::value>
class NTT : public adf::graph {

public:
  static constexpr int NTT_MODULUS_ = MODULUS;
  static constexpr int NTT_ROOT_ = ROOT;
  static constexpr int NTT_LENGTH = LENGTH;
  static constexpr int LENGTH_MOD_INV = ExtendedGCD<LENGTH, MODULUS>::value;
  static constexpr int LAYERS = NO_OF_LAYERS;
  static constexpr int MAX_NO_OF_LAYERS = DETERMINE_LOG_ROOT_SIZE<MODULUS, ROOT>::value;
  static constexpr bool INVERSE_TRANSFORM = is_inverse_transform(TYPE);
  static constexpr NTT_KERNEL_TYPE KERNEL_TYPE = INVERSE_TRANSFORM ? NTT_KERNEL_TYPE::GS_BO_to_NO : NTT_KERNEL_TYPE::CT_NO_to_BO;
  static constexpr int K_BARRETT = 16;
  typedef int16 DATA_TYPE;

private:
  kernel layers[NO_OF_LAYERS];

public:
    input_port in_data;
    output_port out_data;



  NTT(){

    if constexpr (INVERSE_TRANSFORM) {
        // Inverse: separate twiddle factor and Barrett multiplier buffers per layer.
        // The kernel class is templated on the layer index (first/last layer are specialised),
        // so the kernels are constructed with a compile-time index sequence (see auxiliary.h).
        std::vector<std::vector<int16>> twiddle_factors(NO_OF_LAYERS);
        std::vector<std::vector<int16>> barrett_multipliers(NO_OF_LAYERS);
        determine_twiddle_and_barrett<DATA_TYPE, LENGTH, MODULUS, ROOT, K_BARRETT, NTT_KERNEL_TYPE::CT_NO_to_BO, INVERSE_TRANSFORM, NO_OF_LAYERS, MAX_NO_OF_LAYERS>(twiddle_factors, barrett_multipliers);
        construct_layer_kernels_regular<NTT_CT_BO_TO_NO_IMPROVED_KERNEL_CLASS, NO_OF_LAYERS, LENGTH, MODULUS, 16>(layers, twiddle_factors, barrett_multipliers);
    } else {
        // Forward: one combined buffer per layer, [twiddle factors | Barrett multipliers]
        std::vector<std::vector<int16>> factors(NO_OF_LAYERS);
        determine_factors<DATA_TYPE, LENGTH, MODULUS, ROOT, K_BARRETT, KERNEL_TYPE, INVERSE_TRANSFORM, NO_OF_LAYERS, MAX_NO_OF_LAYERS>(factors);
        for (unsigned int i = 0; i < NO_OF_LAYERS; i++) {
            layers[i] = kernel::create_object<NTT_CT_NO_TO_BO_IMPROVED_KERNEL_CLASS<LENGTH, MODULUS, 16>>(factors[i], std::vector<int16>());
        }
    }


    // Layer kernels: one tile per layer
    for (unsigned int i = 0; i < NO_OF_LAYERS; i++) {
        const int row = STARTROW;
        const int col = STARTCOL + i;

        source(layers[i]) = "kernels/ntt_r2_kernels.cc";
        initialization_function(layers[i]) = "set_modes_ntt_layer";
        runtime<ratio>(layers[i]) = 1.0;
        location<kernel>(layers[i]) = tile(col, row);
        if constexpr (KERNEL_TYPE == NTT_KERNEL_TYPE::CT_NO_to_BO) {
            // place the temporary buffer in the same bank as the stack
            location<parameter>(layers[i].param[1]) = location<stack>(layers[i]);
        }

        dimensions(layers[i].in[0]) = { LENGTH };
        dimensions(layers[i].out[0]) = { LENGTH };
    }



    connect(in_data, layers[0].in[0]);


    // Connection between layers
    for (unsigned int i = 0; i < NO_OF_LAYERS-1; i++) {
        connect(layers[i].out[0], layers[i+1].in[0]);
    }


    connect(layers[NO_OF_LAYERS-1].out[0], out_data);


} };
