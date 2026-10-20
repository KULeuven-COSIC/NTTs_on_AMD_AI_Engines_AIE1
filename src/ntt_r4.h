#pragma once

#include <adf.h>
#include "kernels/kernels.h"
#include "kernels/debug.h"
#include "auxiliary.h"
#include "twiddle.h"

using namespace adf;

/*
 * Radix-4 int16 NTT pipeline (Section 5.2 of the paper).
 *
 * log4(n) tiles, one radix-4 layer per tile (constant geometry: every layer reads the four quarters
 * of the buffer and writes the outputs of every 4-NTT consecutively). Requires an even number of
 * radix-2 layers and MODULUS < 2^15 / 3 (lazy reduction of the sum of four partially reduced values).
 *
 * Only the forward transform is implemented as a graph; as described in Section 5.2 the inverse
 * transform is obtained by applying the bit-reversal unit (Section 6, kernels/bitreverse_kernels.cc)
 * and then the forward pipeline with inverse twiddle factors.
 */

enum class NTT_R4_TYPE {
    FORWARD_CT_NO_TO_BO,    // forward NTT, natural-order input, bit-reversed output
};



/* NTT_R4 GRAPH CLASS */
template <unsigned LENGTH, int16 MODULUS, int16 ROOT, unsigned STARTROW, unsigned STARTCOL, NTT_R4_TYPE TYPE, unsigned NO_OF_LAYERS=DETERMINE_LOG_ROOT_SIZE<MODULUS, ROOT>::value>
class NTT_R4 : public adf::graph {

public: 
  static constexpr int NTT_MODULUS_ = MODULUS;
  static constexpr int NTT_ROOT_ = ROOT;
  static constexpr int NTT_LENGTH = LENGTH;
  static constexpr int LENGTH_MOD_INV = ExtendedGCD<LENGTH, MODULUS>::value;
  static constexpr int LAYERS = NO_OF_LAYERS;
  static constexpr int MAX_NO_OF_LAYERS = DETERMINE_LOG_ROOT_SIZE<MODULUS, ROOT>::value;
  static constexpr int NO_OF_R4_LAYERS = int(NO_OF_LAYERS / 2);
  static constexpr int ROOT_SIZE = DETERMINE_ROOT_SIZE<MODULUS, ROOT>::value;
  static constexpr bool INVERSE_TRANSFORM = false;
  static constexpr NTT_KERNEL_TYPE KERNEL_TYPE = NTT_KERNEL_TYPE::CT_NO_to_BO;
  static constexpr int COLS = NO_OF_R4_LAYERS;
  static constexpr unsigned RADIX = 4;
  static constexpr unsigned ROOT_4 = Power<ROOT, int(ROOT_SIZE / RADIX), MODULUS>::value; // primitive 4-th root of unity omega_4
  static constexpr unsigned VECTOR_SIZE = 16;
  static constexpr unsigned K_BARRETT = 16;
  typedef int16 DATA_TYPE;
  
private:
  kernel layers[NO_OF_R4_LAYERS];

public:
  input_port in_data;
  output_port out_data;



  NTT_R4(){

    static_assert(NO_OF_LAYERS % 2 == 0, "MUST BE EVEN AMOUNT OF LAYERS");
    static_assert(MAX_NO_OF_LAYERS % 2 == 0, "MUST BE EVEN AMOUNT OF LAYERS");



    // Determine the factors
    std::vector<std::vector<DATA_TYPE>> factors(NO_OF_R4_LAYERS);
    determine_factors_radix4<DATA_TYPE, LENGTH, MODULUS, ROOT, ROOT_SIZE, K_BARRETT, KERNEL_TYPE, INVERSE_TRANSFORM, NO_OF_R4_LAYERS>(factors);

    // Layer kernels
    for (unsigned int i = 0; i < NO_OF_R4_LAYERS; i++) {   
        const int row = STARTROW;
        const int col = STARTCOL + i;

        layers[i] = kernel::create_object<NTT_R4_CT_NO_TO_BO_KERNEL_CLASS<LENGTH, MODULUS, ROOT_4, VECTOR_SIZE>>(factors[i], std::vector<DATA_TYPE>());


        source(layers[i]) = "kernels/ntt_r4_kernels.cc";
        initialization_function(layers[i]) = "set_modes_ntt_layer";
        runtime<ratio>(layers[i]) = 1.0;
        location<kernel>(layers[i]) = tile(col, row);
        location<parameter>(layers[i].param[1]) = location<stack>(layers[i]);

        dimensions(layers[i].in[0]) = { LENGTH };
        dimensions(layers[i].out[0]) = { LENGTH };
        

    }


    // connect input
    connect(in_data, layers[0].in[0]);

    // Connection between layers
    for (unsigned int i = 0; i < NO_OF_R4_LAYERS-1; i++) {  
        connect(layers[i].out[0], layers[i+1].in[0]);
    }

    // connect output
    connect(layers[NO_OF_R4_LAYERS-1].out[0], out_data);

  }


};
