#pragma once

#include <adf.h>
#include "kernels/kernels.h"
#include "kernels/debug.h"
#include "auxiliary.h"
#include "twiddle.h"

using namespace adf;

/*
 * Radix-2 int32 NTT pipeline (Section 5.1, int32 variant).
 *
 * Same structure as the int16 pipeline in ntt.h: log2(LENGTH) tiles, one radix-2 layer per tile,
 * connected through shared memory. Coefficients are 32-bit (8 lanes per vector), K = 32 for the
 * Barrett multipliers. Constraints: LENGTH <= 2048 and MODULUS < 2^32 / 3 (Table 3).
 */

enum class NTT_INT32_TYPE {
    FORWARD_CT_NO_TO_BO,    // forward NTT, natural-order input, bit-reversed output
    INVERSE_GS_BO_TO_NO,    // inverse NTT, bit-reversed input, natural-order output
};



/* NTT_INT32 GRAPH CLASS */
template <unsigned LENGTH, int32 MODULUS, int32 ROOT, unsigned STARTROW, unsigned STARTCOL, NTT_INT32_TYPE TYPE, unsigned NO_OF_LAYERS=DETERMINE_LOG_ROOT_SIZE<MODULUS, ROOT>::value>
class NTT_INT32 : public adf::graph {

public: 
  static constexpr int NTT_MODULUS_ = MODULUS;
  static constexpr int NTT_ROOT_ = ROOT;
  static constexpr int NTT_LENGTH = LENGTH;
  static constexpr int LENGTH_MOD_INV = ExtendedGCD<LENGTH, MODULUS>::value;
  static constexpr int LAYERS = NO_OF_LAYERS;
  static constexpr int MAX_NO_OF_LAYERS = DETERMINE_LOG_ROOT_SIZE<MODULUS, ROOT>::value;
  static constexpr int FACTORS_LENGTH = (1 << (NO_OF_LAYERS - 1)); // 2^{NO_OF_LAYERS-1}
  static constexpr bool IS_GS_BUTTERFLY = TYPE == NTT_INT32_TYPE::INVERSE_GS_BO_TO_NO;
  static constexpr bool INVERSE_TRANSFORM = TYPE == NTT_INT32_TYPE::INVERSE_GS_BO_TO_NO;
  static constexpr NTT_KERNEL_TYPE KERNEL_TYPE = (TYPE == NTT_INT32_TYPE::FORWARD_CT_NO_TO_BO) ? NTT_KERNEL_TYPE::CT_NO_to_BO : NTT_KERNEL_TYPE::GS_BO_to_NO;
  static constexpr int COLS = NO_OF_LAYERS;
  static constexpr unsigned VECTOR_SIZE = 8;
  static constexpr unsigned K_BARRETT = 32;
  typedef int32 DATA_TYPE;
  
private:
  kernel layers[NO_OF_LAYERS];

public:
  input_port in_data;
  output_port out_data;



  NTT_INT32(){



    // Determine the factors
    std::vector<std::vector<DATA_TYPE>> factors(NO_OF_LAYERS);
    determine_factors<DATA_TYPE, LENGTH, MODULUS, ROOT, K_BARRETT, KERNEL_TYPE, INVERSE_TRANSFORM, NO_OF_LAYERS, MAX_NO_OF_LAYERS>(factors);

    // Layer kernels
    for (unsigned int i = 0; i < NO_OF_LAYERS; i++) {   
        const int row = STARTROW;
        const int col = STARTCOL + i;

        if constexpr (KERNEL_TYPE == NTT_KERNEL_TYPE::CT_NO_to_BO) {
            layers[i] = kernel::create_object<NTT_INT32_CT_NO_TO_BO_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE>>(factors[i], std::vector<DATA_TYPE>());
        } else {
            layers[i] = kernel::create_object<NTT_INT32_GS_BO_TO_NO_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE>>(factors[i], std::vector<DATA_TYPE>());
        }


        source(layers[i]) = "kernels/ntt_r2_int32_kernels.cc";
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
    for (unsigned int i = 0; i < NO_OF_LAYERS-1; i++) {  
        connect(layers[i].out[0], layers[i+1].in[0]);
    }

    // connect output
    connect(layers[NO_OF_LAYERS-1].out[0], out_data);

  }


};