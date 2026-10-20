#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "aie_api/types.hpp"
#include "debug.h"
#include "kernels.h"

#ifdef __X86SIM__
#include <adf/x86sim/x86simDebug.h>
#endif

/*
 * Radix-8 int16 NTT kernels (Section 5.3 of the paper). One radix-8 layer is split over two tiles:
 *
 *   NTT_R8_LAYERS    (block 1): radix-8 butterflies on 16-bit inputs, computed as 8x8 matrix-vector
 *                               products with mul8/mac8 (first four inputs, then last four inputs) and
 *                               left unreduced, so the outputs are 32-bit.
 *   NTT_R8_REDUCTION (block 2): 32-to-16-bit Barrett multiplication with the between-layer twiddle
 *                               factors, in three loops: (i) m' = a * w' (32x32-bit, K = 31),
 *                               (ii) truncate m' to 16 bits, (iii) a * w - m' * q, fused with the
 *                               permutation that de-interleaves and bit-reverses the 8-NTT outputs.
 */



template<unsigned MODULUS, unsigned LENGTH, unsigned VECTOR_SIZE>
NTT_R8_RECUTION_KERNEL_CLASS<MODULUS, LENGTH, VECTOR_SIZE>::NTT_R8_RECUTION_KERNEL_CLASS(int32 (&barrett_factors_)[LENGTH], int16 (&twiddle_factors_)[LENGTH], int16 (&temp_buffer_)[LENGTH], int32 (&temp_buffer2_)[LENGTH>>1]) : barrett_factors(barrett_factors_), twiddle_factors(twiddle_factors_), temp_buffer(temp_buffer_), temp_buffer2(temp_buffer2_) {}


template<unsigned MODULUS, unsigned LENGTH, unsigned VECTOR_SIZE>
__attribute__((noinline)) void NTT_R8_RECUTION_KERNEL_CLASS<MODULUS, LENGTH, VECTOR_SIZE>::NTT_R8_REDUCTION(
        adf::input_buffer<int32> & __restrict in_data,
        adf::output_buffer<int16> & __restrict out_data) {

    // Get the barret factor for 1
    aie::tile tile_obj=aie::tile::current(); //get the tile of the kernel
    tile_obj.set_rounding(aie::rounding_mode::symmetric_zero);
    constexpr int K_mult = 31; // Must be this high to be able to reduce products of 30bits
    constexpr unsigned IT = LENGTH / VECTOR_SIZE;

  
    {
        // The in and out buffers switch every layer
        int32 __aie_dm_resource_a* __restrict cur_layer_in_data_start = (int32 __aie_dm_resource_a*)in_data.data();
        auto firstTermVectorIt = aie::begin_vector<VECTOR_SIZE/2>(cur_layer_in_data_start);


        int32 __aie_dm_resource_b* __restrict barrett_factors_start = (int32 __aie_dm_resource_b*) barrett_factors;
        auto barret_factors_it = aie::begin_vector<VECTOR_SIZE/2>(barrett_factors_start);

        int32 __aie_dm_resource_c* __restrict out_buffer_start = (int32 __aie_dm_resource_c*) out_data.data();
        auto vectorOUTIt = aie::begin_vector<VECTOR_SIZE/2>(out_buffer_start);

        int32 __aie_dm_resource_d* __restrict temp_buffer2_start = (int32 __aie_dm_resource_d*) temp_buffer2;
        auto vectorOUT2It = aie::begin_vector<VECTOR_SIZE/2>(temp_buffer2_start);
        
        constexpr unsigned K_mult = 31;
        // 1 cc per 8 samples
        for (unsigned j=0; j < IT/2; j++) 
            chess_prepare_for_pipelining { 
                // manually unroll four times because has to write out to two different banks
                {
            aie::vector<int32, VECTOR_SIZE/2> muls = *firstTermVectorIt++;
            aie::vector<int32, VECTOR_SIZE/2> barret_factors = *barret_factors_it++;
                LOG_VECTOR(muls, true, "muls=");
            aie::accum<acc80, VECTOR_SIZE/2> mul_barret = aie::mul(muls, barret_factors);
                LOG_VECTOR(mul_barret, true, "mul_barret=");
            aie::vector<int32, VECTOR_SIZE/2> mul_barret_vec = mul_barret.template to_vector<int32>(K_mult);
                LOG_VECTOR(mul_barret_vec, true, "mul_barret_vec=");
            *vectorOUTIt++ = mul_barret_vec;
                }

                {
            aie::vector<int32, VECTOR_SIZE/2> muls = *firstTermVectorIt++;
            aie::vector<int32, VECTOR_SIZE/2> barret_factors = *barret_factors_it++;
                LOG_VECTOR(muls, true, "muls=");
            aie::accum<acc80, VECTOR_SIZE/2> mul_barret = aie::mul(muls, barret_factors);
                LOG_VECTOR(mul_barret, true, "mul_barret=");
            aie::vector<int32, VECTOR_SIZE/2> mul_barret_vec = mul_barret.template to_vector<int32>(K_mult);
                LOG_VECTOR(mul_barret_vec, true, "mul_barret_vec=");
            *vectorOUTIt++ = mul_barret_vec;
                }

                {
            aie::vector<int32, VECTOR_SIZE/2> muls = *firstTermVectorIt++;
            aie::vector<int32, VECTOR_SIZE/2> barret_factors = *barret_factors_it++;
                LOG_VECTOR(muls, true, "muls=");
            aie::accum<acc80, VECTOR_SIZE/2> mul_barret = aie::mul(muls, barret_factors);
                LOG_VECTOR(mul_barret, true, "mul_barret=");
            aie::vector<int32, VECTOR_SIZE/2> mul_barret_vec = mul_barret.template to_vector<int32>(K_mult);
                LOG_VECTOR(mul_barret_vec, true, "mul_barret_vec=");
            *vectorOUT2It++ = mul_barret_vec;
                }

                {
            aie::vector<int32, VECTOR_SIZE/2> muls = *firstTermVectorIt++;
            aie::vector<int32, VECTOR_SIZE/2> barret_factors = *barret_factors_it++;
                LOG_VECTOR(muls, true, "muls=");
            aie::accum<acc80, VECTOR_SIZE/2> mul_barret = aie::mul(muls, barret_factors);
                LOG_VECTOR(mul_barret, true, "mul_barret=");
            aie::vector<int32, VECTOR_SIZE/2> mul_barret_vec = mul_barret.template to_vector<int32>(K_mult);
                LOG_VECTOR(mul_barret_vec, true, "mul_barret_vec=");
            *vectorOUT2It++ = mul_barret_vec;
                }
        }
    }

    {
        int16 __aie_dm_resource_b* __restrict temp_buffer_start = (int16 __aie_dm_resource_b*) temp_buffer;
        auto lastTermVectorIt = aie::begin_vector<VECTOR_SIZE>(temp_buffer_start);

        int32 __aie_dm_resource_c* __restrict out_buffer_start = (int32 __aie_dm_resource_c*) out_data.data();
        auto vectorOUTIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_start);

        int32 __aie_dm_resource_d* __restrict temp_buffer2_start = (int32 __aie_dm_resource_d*) temp_buffer2;
        auto vectorOUT2It = aie::begin_vector<VECTOR_SIZE>(temp_buffer2_start);
        

        // 1 cc per 16 samples
     for (unsigned j=0; j < IT/2; j++) 
            chess_prepare_for_pipelining { 
            aie::vector<int32, VECTOR_SIZE> muls = *vectorOUTIt++;
            aie::accum<acc48, VECTOR_SIZE> acc = aie::mul(muls, int16(1));
            *lastTermVectorIt++ = acc.template to_vector<int16>(0);
            aie::vector<int32, VECTOR_SIZE> muls2 = *vectorOUT2It++;
            aie::accum<acc48, VECTOR_SIZE> acc2 = aie::mul(muls2, int16(1));
            *lastTermVectorIt++ = acc2.template to_vector<int16>(0);


                LOG_VECTOR(muls, true, "mul_barret_vec=");
                LOG_VECTOR(acc.template to_vector<int16>(0), true, "mul_barret_vec as 16b=");
                LOG_VECTOR(muls2, true, "mul_barret_vec=");
                LOG_VECTOR(acc2.template to_vector<int16>(0), true, "mul_barret_vec as 16b=");
        }
    }

    {
        int16 __aie_dm_resource_a* __restrict temp_buffer_start = (int16 __aie_dm_resource_a*) temp_buffer;
        auto lastTermVectorIt = aie::begin_vector<VECTOR_SIZE>(temp_buffer_start);
        
        int32 __aie_dm_resource_d* __restrict cur_layer_in_data_start = (int32 __aie_dm_resource_d*)in_data.data();
        auto firstTermVectorIt = aie::begin_vector<VECTOR_SIZE>(cur_layer_in_data_start);

        int16 __aie_dm_resource_b* __restrict twiddle_factors_start = (int16 __aie_dm_resource_b*) twiddle_factors;
        auto twiddle_factors_it = aie::begin_vector<VECTOR_SIZE>(twiddle_factors_start);

        int16 __aie_dm_resource_c* __restrict out_buffer_start = (int16 __aie_dm_resource_c*) out_data.data();
        auto vectorOUTIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_start);

        // 2 cc per 16 samples
        // make vector with constants such that we can use mac16 for permutation
        // The required permutation: At input, two 8-NTT outputs are interleaved, so (a = first 8-NTT, b = second 8-NTT): a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7
        // They have to be unshuffled AND bit-reversed within each NTT, so: a0,a4,a2,a6,a1,a5,a3,a7,b0,b4,b2,b6,b1,b5,b3,b7
        // This is facilitated by changing both the twiddle multiplication (mul16) and reduction (msc16), such that in the reduction both the accumulators as the coefficients are in the right order
        alignas(16) const int16 modulo_cst[32] = {MODULUS, 0, MODULUS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        v32int16 cst = aie::load_v<2*VECTOR_SIZE>(modulo_cst);
        for (unsigned j=0; j < IT; j++) 
            chess_prepare_for_pipelining { 
            aie::vector<int32, VECTOR_SIZE> muls = *firstTermVectorIt++;
            aie::vector<int16, VECTOR_SIZE> twiddle_factors = *twiddle_factors_it++;
                LOG_VECTOR(twiddle_factors, true, "twiddle_factors=");
            aie::accum<acc48, VECTOR_SIZE> mul_twiddle = mul16(muls, 0, 0xe6a2c480, 0xf7b3d591, twiddle_factors, 0, 0xe6a2c480, 0xf7b3d591); // permutation for bit-reversal inside one 8-NTT output and to combine two interleaved 8-NTTs
                LOG_VECTOR(muls, true, "muls=");
                LOG_VECTOR(mul_twiddle, true, "mul_twiddle=");
            aie::vector<int16, VECTOR_SIZE> mul_barret_vec = *lastTermVectorIt++;
                LOG_VECTOR(mul_barret_vec, true, "mul_barret_vec=");
            v16acc48 mul_twiddle_v = mul_twiddle;
            v16int16 mul_barret_vec_v = mul_barret_vec;
            v16acc48 reduced_acc_v = msc16(mul_twiddle_v, cst, 0,0x00000000,0x00000000,0x3210, mul_barret_vec_v, 0,0xe6a2c480, 0xf7b3d591,0);  // permutation for bit-reversal inside one 8-NTT output and to combine two interleaved 8-NTTs
            aie::accum<acc48, VECTOR_SIZE> reduced_acc = reduced_acc_v; 
                LOG_VECTOR(reduced_acc, true, "reduced_acc=");
                LOG_VECTOR(reduced_acc.template to_vector<int16>(0), true, "reduced_acc as vector=");
            *vectorOUTIt++ = reduced_acc.template to_vector<int16>(0);
        }
    }

}



template<unsigned MODULUS, unsigned LENGTH, unsigned VECTOR_SIZE>
NTT_R8_LAYERS_KERNEL_CLASS<MODULUS, LENGTH, VECTOR_SIZE>::NTT_R8_LAYERS_KERNEL_CLASS(int16 (&twiddle_factors_)[TWIDDLE_FACTORS_LENGTH], int32 (&temp_buffer2_)[LENGTH]) : twiddle_factors(twiddle_factors_), temp_buffer2(temp_buffer2_) {}



#define first_four_terms(input_buffer_permutation) do { \
    v8acc48 chess_storage(aml0) first_four_outputs = mul8(xbuf, 0, input_buffer_permutation, 16, 0x3120, twiddle, 0, 0xcc884400, 1);  \
    *temp2It++ = lsrs(first_four_outputs, 0);                                                       \
        aie::vector<int32, VECTOR_SIZE/2> first_four_outputs_v = lsrs(first_four_outputs, 0);       \
        LOG_VECTOR(first_four_outputs_v, true, "first_four_outputs_v=");                            \
    v8acc48 chess_storage(aml0) last_four_outputs = mul8(xbuf, 0, input_buffer_permutation, 16, 0x3120, twiddle2, 0, 0xcc884400, 1);  \
    *temp2It++ = lsrs(last_four_outputs, 0);                                                        \
        aie::vector<int32, VECTOR_SIZE/2> last_four_outputs_v = lsrs(last_four_outputs, 0);         \
        LOG_VECTOR(last_four_outputs_v, true, "last_four_outputs_v=");                              \
} while(0);



#define last_four_terms(input_buffer_permutation) do { \
    v8int32 first_four_outputs_vec = *temp2It++;                                                                                    \
    v8acc48 chess_storage(aml0) first_four_outputs = mac8(ups(first_four_outputs_vec, 0), xbuf, 0, input_buffer_permutation, 16, 0x3120, twiddle, 0, 0xcc884400, 1);  \
    *outIt++ = lsrs(first_four_outputs, 0);                                                                                         \
        aie::vector<int32, VECTOR_SIZE/2> first_four_outputs_v = lsrs(first_four_outputs, 0);                                       \
        LOG_VECTOR(first_four_outputs_v, true, "first_four_outputs_v=");                                                            \
    v8int32 last_four_outputs_vec = *temp2It++;                                                                                     \
    v8acc48 chess_storage(aml0) last_four_outputs = mac8(ups(last_four_outputs_vec, 0), xbuf, 0, input_buffer_permutation, 16, 0x3120, twiddle2, 0, 0xcc884400, 1);   \
    *outIt++ = lsrs(last_four_outputs, 0);                                                                                          \
        aie::vector<int32, VECTOR_SIZE/2> last_four_outputs_v = lsrs(last_four_outputs, 0);                                         \
        LOG_VECTOR(last_four_outputs_v, true, "last_four_outputs_v=");                                                              \
} while(0);



template<unsigned MODULUS, unsigned LENGTH, unsigned VECTOR_SIZE>
__attribute__((noinline)) void NTT_R8_LAYERS_KERNEL_CLASS<MODULUS, LENGTH, VECTOR_SIZE>::NTT_R8_LAYERS(
        adf::input_buffer<int16> & __restrict in_data,
        adf::output_buffer<int32> & __restrict out_data) {




    // Some constants
    constexpr unsigned SUBNTT_SIZE = 8;
    constexpr unsigned NTT_GROUP_SIZE = 8; // because we use mac8: calculate the same output of 8 NTTs at once
    constexpr unsigned MAC_PER_CYCLE = 4; // 4 MACs per accumulator per mac intrinsic
    constexpr unsigned GROUPS_PER_INPUT_BUFFER = 2; // one xbuffer of inputs can supply this many groups of inputs
    constexpr unsigned NO_OF_SUBNTTS = LENGTH / SUBNTT_SIZE;
    constexpr unsigned NO_NTT_GROUPS = NO_OF_SUBNTTS / NTT_GROUP_SIZE;
    constexpr unsigned DISTANCE_TO_NEXT_INPUT = NO_OF_SUBNTTS / VECTOR_SIZE;

    // first four MACs for every output
    {
        LOG("FIRST FOUR TERMS/MACs/Inputs");

        int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*) in_data.data();
        auto inIt = aie::begin_vector_random_circular<VECTOR_SIZE, LENGTH/2>(in_buffer_start);

        int16 __aie_dm_resource_b* __restrict twiddle_buffer_start = (int16 __aie_dm_resource_b*) twiddle_factors;
        auto twiddleIt = aie::begin_vector_circular<VECTOR_SIZE, SUBNTT_SIZE * MAC_PER_CYCLE>(twiddle_buffer_start);

        int32 __aie_dm_resource_c* __restrict temp2_buffer_start = (int32 __aie_dm_resource_c*) temp_buffer2;
        auto temp2It = aie::begin_vector<VECTOR_SIZE/2>(temp2_buffer_start);


        // Load twiddle factors
        // Contains factors for all 8 outputs of an 8-NTT, but only for the first 4 inputs -> 32 factors
        v16int16 twiddle = *twiddleIt++;
            aie::vector<int16, VECTOR_SIZE> t_v = twiddle;
            LOG_VECTOR(t_v, true, "\nt_v=");
        v16int16 twiddle2 = *twiddleIt++;
            aie::vector<int16, VECTOR_SIZE> t2_v = twiddle2;
            LOG_VECTOR(t2_v, true, "\nt2_v=");

        // no of it: number of 16-NTT sets / NTT_GROUP_SIZE
        // After this loop: four MACs of every output of every NTT set
        for (unsigned i = 0; i < NO_NTT_GROUPS / GROUPS_PER_INPUT_BUFFER; i++) chess_prepare_for_pipelining {
            // Load input vectors (xbuffer)
            // Contains the first 4 inputs for 16 consecutive 8-NTTs -> 64 samples
            v16int16 chess_storage(wr0) input1 = *inIt; inIt += DISTANCE_TO_NEXT_INPUT;
                    aie::vector<int16, VECTOR_SIZE> input1_v = input1;
                    LOG_VECTOR(input1_v, true, "\ninput1_v=");
            v16int16 chess_storage(wr1) input2 = *inIt; inIt += DISTANCE_TO_NEXT_INPUT;
                    aie::vector<int16, VECTOR_SIZE> input2_v = input2;
                    LOG_VECTOR(input2_v, true, "\ninput2_v=");
            v16int16 chess_storage(wr2) input3 = *inIt; inIt += DISTANCE_TO_NEXT_INPUT;
                    aie::vector<int16, VECTOR_SIZE> input3_v = input3;
                    LOG_VECTOR(input3_v, true, "\ninput3_v=");
            v16int16 chess_storage(wr3) input4 = *inIt; inIt += DISTANCE_TO_NEXT_INPUT + 1;
                    aie::vector<int16, VECTOR_SIZE> input4_v = input4;
                    LOG_VECTOR(input4_v, true, "\ninput4_v=");
            v64int16 xbuf = concat(input1, input2, input3, input4);
            


            // We now have the inputs and factors to calculate the first four terms for all 8 outputs of 16 consecutive 8-NTTs
            // This is done in 8 groups of two 8-NTTs (the outputs are interleaved), becuase of permutation constraints of mul8
            // The only difference between the 8 groups is the permutation of the xbuffer

            // Calculate first four terms (MACs/inputs) for all 8 outputs of the first two 8-NTTs
            // The outputs of the two 8-NTTs are interleaved (due to permutation constraints of mul8)
            LOG("\nFIRST FOUR TERMS: 8-NTT SET 1/8\n");
            first_four_terms(0xf0f0f0f0);

            // Second set of two 8-NTTs  
            LOG("\nFIRST FOUR TERMS: 8-NTT SET 2/8\n");
            first_four_terms(0xf1f1f1f1);
                   
            // Third set of two 8-NTTs  
            LOG("\nFIRST FOUR TERMS: 8-NTT SET 3/8\n");
            first_four_terms(0xf2f2f2f2);
             
            // Fourth set of two 8-NTTs  
            LOG("\nFIRST FOUR TERMS: 8-NTT SET 4/8\n");
            first_four_terms(0xf3f3f3f3);

            // Fifth set of two 8-NTTs  
            LOG("\nFIRST FOUR TERMS: 8-NTT SET 5/8\n");
            first_four_terms(0xf4f4f4f4);

            // Sixth set of two 8-NTTs  
            LOG("\nFIRST FOUR TERMS: 8-NTT SET 6/8\n");
            first_four_terms(0xf5f5f5f5);
                   
            // Seventh set of two 8-NTTs  
            LOG("\nFIRST FOUR TERMS: 8-NTT SET 7/8\n");
            first_four_terms(0xf6f6f6f6);
             
            // Eight set of two 8-NTTs  
            LOG("\nFIRST FOUR TERMS: 8-NTT SET 8/8\n");
            first_four_terms(0xf7f7f7f7);
               
             

        }
    }

      // last four terms for every output
      
    {
        LOG("LAST FOUR TERMS/MACs/Inputs");

        int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*) in_data.data() + (LENGTH>>1);
        auto inIt = aie::begin_vector_random_circular<VECTOR_SIZE, LENGTH/2>(in_buffer_start);

        int32 __aie_dm_resource_d* __restrict out_buffer_start = (int32 __aie_dm_resource_d*) out_data.data();
        auto outIt = aie::begin_vector_random_circular<VECTOR_SIZE/2, LENGTH>(out_buffer_start);

        int16 __aie_dm_resource_b* __restrict twiddle_buffer_start = (int16 __aie_dm_resource_b*) twiddle_factors + (SUBNTT_SIZE * MAC_PER_CYCLE);
        auto twiddleIt = aie::begin_vector_circular<VECTOR_SIZE, SUBNTT_SIZE * MAC_PER_CYCLE>(twiddle_buffer_start);

        int32 __aie_dm_resource_c* __restrict temp2_buffer_start = (int32 __aie_dm_resource_c*) temp_buffer2;
        auto temp2It = aie::begin_vector<VECTOR_SIZE/2>(temp2_buffer_start);


        // Load twiddle factors
        // Contains factors for all 8 outputs of an 8-NTT, but only for the last 4 inputs -> 32 factors
        v16int16 twiddle = *twiddleIt++;
            aie::vector<int16, VECTOR_SIZE> t_v = twiddle;
            LOG_VECTOR(t_v, true, "\nt_v=");
        v16int16 twiddle2 = *twiddleIt++;
            aie::vector<int16, VECTOR_SIZE> t2_v = twiddle2;
            LOG_VECTOR(t2_v, true, "\nt2_v=");

        for (unsigned i = 0; i < NO_NTT_GROUPS / GROUPS_PER_INPUT_BUFFER; i++) chess_prepare_for_pipelining {
            v16int16 chess_storage(wr0) input1 = *inIt; inIt += DISTANCE_TO_NEXT_INPUT;
                    aie::vector<int16, VECTOR_SIZE> input1_v = input1;
                    LOG_VECTOR(input1_v, true, "\ninput1_v=");
            v16int16 chess_storage(wr1) input2 = *inIt; inIt += DISTANCE_TO_NEXT_INPUT;
                    aie::vector<int16, VECTOR_SIZE> input2_v = input2;
                    LOG_VECTOR(input2_v, true, "\ninput2_v=");
            v16int16 chess_storage(wr2) input3 = *inIt; inIt += DISTANCE_TO_NEXT_INPUT;
                    aie::vector<int16, VECTOR_SIZE> input3_v = input3;
                    LOG_VECTOR(input3_v, true, "\ninput3_v=");
            v16int16 chess_storage(wr3) input4 = *inIt; inIt += DISTANCE_TO_NEXT_INPUT + 1;
                    aie::vector<int16, VECTOR_SIZE> input4_v = input4;
                    LOG_VECTOR(input4_v, true, "\ninput4_v=");
            v64int16 xbuf = concat(input1, input2, input3, input4);

            


            // We now have the inputs and factors to calculate the last four terms for all 8 outputs of 16 consecutive 8-NTTs
            // This is done in 8 groups of two 8-NTTs (the outputs are interleaved), becuase of permutation constraints of mul8
            // The only difference between the 8 groups is the permutation of the xbuffer

            // Calculate last four terms (MACs/inputs) for all 8 outputs of the first two 8-NTTs
            // The outputs of the two 8-NTTs are interleaved (due to permutation constraints of mul8)
            LOG("\nLAST FOUR TERMS: 8-NTT SET 1/8\n");
            last_four_terms(0xf0f0f0f0);

            // Second set of two 8-NTTs  
            LOG("\nLAST FOUR TERMS: 8-NTT SET 2/8\n");
            last_four_terms(0xf1f1f1f1);
                   
            // Third set of two 8-NTTs  
            LOG("\nLAST FOUR TERMS: 8-NTT SET 3/8\n");
            last_four_terms(0xf2f2f2f2);
             
            // Fourth set of two 8-NTTs  
            LOG("\nLAST FOUR TERMS: 8-NTT SET 4/8\n");
            last_four_terms(0xf3f3f3f3);

            // Fifth set of two 8-NTTs  
            LOG("\nLAST FOUR TERMS: 8-NTT SET 5/8\n");
            last_four_terms(0xf4f4f4f4);

            // Sixth set of two 8-NTTs  
            LOG("\nLAST FOUR TERMS: 8-NTT SET 6/8\n");
            last_four_terms(0xf5f5f5f5);
                   
            // Seventh set of two 8-NTTs  
            LOG("\nLAST FOUR TERMS: 8-NTT SET 7/8\n");
            last_four_terms(0xf6f6f6f6);
             
            // Eight set of two 8-NTTs  
            LOG("\nLAST FOUR TERMS: 8-NTT SET 8/8\n");
            last_four_terms(0xf7f7f7f7);

             
        }

    }

}