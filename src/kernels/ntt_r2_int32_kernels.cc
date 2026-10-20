

#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "aie_api/iterator.hpp"
#include "aie_api/types.hpp"
#include "debug.h"
#include "kernels.h"

#ifdef __X86SIM__
#include <adf/x86sim/x86simDebug.h>
#endif

/*
 * Radix-2 int32 NTT layer kernels (Section 5.1, int32 variant; Barrett primitives of Section 4.2).
 *
 * With 8 lanes of int32 and 8 32x32-bit multiplications per cycle, a partial Barrett multiplication
 * needs three MAC operations. A single loop does not reach II = 3, so it is split (loop fission) into:
 *   loop 1 (Figure 3a): m' = round(a * w' / 2^32)           -> *_barrett_multiplier_multiplication, II = 1
 *   loop 2 (Figure 3c): r  = a * w - m' * q                 -> *_twiddle_multiplication_and_reduction, II = 2
 * For a Barrett reduction (w = 1) the second loop only needs one MAC (Figure 3b, II = 1):
 *   ntt_int32_constant_barrett_multiplier_multiplication + ntt_int32_constant_twiddle_multiplication_and_reduction.
 * The butterflies (add/sub, left unreduced) are fused with the (un)shuffle permutation using lmul8
 * with a +-1 constant vector (32x16-bit multiplications).
 */




template<unsigned LENGTH, unsigned MODULUS, unsigned K_BARRETT, unsigned VECTOR_SIZE, unsigned ACTUAL_FACTORS_LENGTH>
inline void ntt_int32_barrett_multiplier_multiplication(int32* const operands, int32* const barrett_multipliers, int32* const results_out) {
    LOG_STATUS("barrett_multiplier_multiplication:\n");

    int32 __aie_dm_resource_d* __restrict temp_buffer_start = (int32 __aie_dm_resource_d*) results_out;
    auto tempIt = aie::begin_vector<VECTOR_SIZE>(temp_buffer_start);

    int32 __aie_dm_resource_c* __restrict factors_buffer_middle = (int32 __aie_dm_resource_c*) barrett_multipliers;
    auto factorsMiddleIt = aie::cbegin_vector_circular<VECTOR_SIZE, ACTUAL_FACTORS_LENGTH>(factors_buffer_middle);

    int32 __aie_dm_resource_a* __restrict in_buffer_middle = (int32 __aie_dm_resource_a*) operands;
    auto inMiddleIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_middle);

    // 1cc per iteration
    for (unsigned j=0; j < LENGTH / VECTOR_SIZE; j++) {
        aie::vector<int32, VECTOR_SIZE> sub = *inMiddleIt++;
        aie::vector<int32, VECTOR_SIZE> b = *factorsMiddleIt++;
            LOG_VECTOR(sub, true, "\nv1=");
            LOG_VECTOR(b, true, "\nb=");
        aie::accum<acc80, VECTOR_SIZE> mb = aie::mul(sub,b);
        aie::vector<int32, VECTOR_SIZE> mb_vec = mb.template to_vector<int32>(K_BARRETT);
            LOG_VECTOR(mb_vec, true, "\nmb_vec=");
        *tempIt++ = mb_vec;
    }
}

template<unsigned LENGTH, unsigned MODULUS, unsigned K_BARRETT, unsigned VECTOR_SIZE, unsigned ACTUAL_FACTORS_LENGTH>
inline void ntt_int32_twiddle_multiplication_and_reduction(int32* const operands, int32* const twiddle_factors, int32* const barrett_multiplications, int32* const results_out) {
    LOG_STATUS("twiddle_multiplication_and_reduction:\n");

    int32 __aie_dm_resource_a* __restrict in_buffer_middle = (int32 __aie_dm_resource_a*) operands;
    auto inMiddleIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_middle);
    
    int32 __aie_dm_resource_a* __restrict out_buffer_middle = (int32 __aie_dm_resource_a*) results_out;
    auto outMiddleIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_middle);

    int32 __aie_dm_resource_c* __restrict factors_buffer_start = (int32 __aie_dm_resource_c*) twiddle_factors;
    auto factorsIt = aie::cbegin_vector_circular<VECTOR_SIZE, ACTUAL_FACTORS_LENGTH>(factors_buffer_start);

    int32 __aie_dm_resource_d* __restrict temp_buffer_start = (int32 __aie_dm_resource_d*) barrett_multiplications;
    auto tempIt = aie::begin_vector<VECTOR_SIZE>(temp_buffer_start);

    // 2cc per iteration
    for (unsigned j=0; j < LENGTH / VECTOR_SIZE; j++) {
        aie::vector<int32, VECTOR_SIZE> sub = *inMiddleIt++;
        aie::vector<int32, VECTOR_SIZE> t = *factorsIt++;
        aie::vector<int32, VECTOR_SIZE> mb_vec = *tempIt++;
        aie::accum<acc80, VECTOR_SIZE> mt = aie::mul(sub,t);
            LOG_VECTOR(sub, true, "\nsub=");
            LOG_VECTOR(t, true, "\nt=");
            LOG_VECTOR(mb_vec, true, "\nmb_vec=");
            LOG_VECTOR(mt.template to_vector<int32>(0), true, "\nmt as vec=");
        aie::accum<acc80, VECTOR_SIZE> red_acc = aie::msc(mt,mb_vec, (int32)MODULUS);
            LOG_VECTOR(red_acc.template to_vector<int32>(0), true, "\nred as vec=");
        v8int32 chess_storage(wr2) red_vec = red_acc.template to_vector<int32>(0); // Has to be stored in a vector after SRSS (don't use VST.SRSS), because if we use VST.SRSS and write to a bank where we are reading from then there will be memory stalls due to a compiler bug
        *outMiddleIt++ = red_vec;
    }
}


template<unsigned LENGTH, unsigned MODULUS, unsigned K_BARRETT, unsigned VECTOR_SIZE, unsigned ACTUAL_FACTORS_LENGTH>
inline void ntt_int32_constant_barrett_multiplier_multiplication(int32* const operands, int32* const results_out) {
    LOG_STATUS("constant_barrett_multiplier_multiplication:\n");

    int32 __aie_dm_resource_d* __restrict temp_buffer_start = (int32 __aie_dm_resource_d*) results_out;
    auto tempIt = aie::begin_vector<VECTOR_SIZE>(temp_buffer_start);

    int32 __aie_dm_resource_a* __restrict in_buffer_middle = (int32 __aie_dm_resource_a*) operands;
    auto inMiddleIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_middle);

    // 1cc per iteration
    const int32 barrett_factor = int32((long double)((long long)(1) << K_BARRETT) / (long double)(MODULUS) + (long double)0.5);
    for (unsigned j=0; j < LENGTH / VECTOR_SIZE; j++) {
        aie::vector<int32, VECTOR_SIZE> sub = *inMiddleIt++;
            LOG_VECTOR(sub, true, "\nv1=");
        aie::accum<acc80, VECTOR_SIZE> mb = aie::mul(sub,barrett_factor);
        aie::vector<int32, VECTOR_SIZE> mb_vec = mb.template to_vector<int32>(K_BARRETT);
            LOG_VECTOR(mb_vec, true, "\nmb_vec=");
        *tempIt++ = mb_vec;
    }
}

template<unsigned LENGTH, unsigned MODULUS, unsigned K_BARRETT, unsigned VECTOR_SIZE, unsigned ACTUAL_FACTORS_LENGTH>
inline void ntt_int32_constant_twiddle_multiplication_and_reduction(int32* const operands, int32* const barrett_multiplications, int32* const results_out) {
    LOG_STATUS("constant_twiddle_multiplication_and_reduction:\n");

    int32 __aie_dm_resource_a* __restrict in_buffer_middle = (int32 __aie_dm_resource_a*) operands;
    auto inMiddleIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_middle);
    
    int32 __aie_dm_resource_b* __restrict out_buffer_middle = (int32 __aie_dm_resource_b*) results_out;
    auto outMiddleIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_middle);

    int32 __aie_dm_resource_d* __restrict temp_buffer_start = (int32 __aie_dm_resource_d*) barrett_multiplications;
    auto tempIt = aie::begin_vector<VECTOR_SIZE>(temp_buffer_start);

    // 2cc per iteration
    for (unsigned j=0; j < LENGTH / VECTOR_SIZE; j++) {
        aie::vector<int32, VECTOR_SIZE> mb_vec = *tempIt++;
        aie::vector<int32, VECTOR_SIZE> mt = *inMiddleIt++;
            LOG_VECTOR(mt, true, "\nT=");
            LOG_VECTOR(mb_vec, true, "\nmb_vec=");
        aie::accum<acc80, VECTOR_SIZE> mul_acc = aie::msc(aie::accum<acc80, VECTOR_SIZE>(mt),mb_vec, (int32)MODULUS);
            LOG_VECTOR(mul_acc.template to_vector<int32>(0), true, "\nreduced as vec=");
        *outMiddleIt++ = mul_acc.template to_vector<int32>(0);
    }
}

template<unsigned LENGTH, unsigned VECTOR_SIZE>
inline void ntt_int32_CT_add_sub_interleave(int32* const input_low, int32* const input_high, int32* output) {
    LOG_STATUS("CT_add_sub_interleav:\n");

    int32 __aie_dm_resource_a* __restrict in_buffer_start = (int32 __aie_dm_resource_a*) input_low;
    auto inIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_start);

    int32 __aie_dm_resource_a* __restrict in_buffer_start2 = (int32 __aie_dm_resource_a*) input_high;
    auto inIt2 = aie::begin_vector<VECTOR_SIZE>(in_buffer_start2);

    int32 __aie_dm_resource_b* __restrict out_buffer_start = (int32 __aie_dm_resource_b*) output;
    auto outIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_start);

    alignas(16) const int16 cst_arr[16] = {1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   
    aie::vector<int16,2*VECTOR_SIZE> cst = aie::load_v<2*VECTOR_SIZE>(cst_arr); 
    for (unsigned j=0; j < LENGTH / (4 * VECTOR_SIZE); j++) {
        // first iteration
        aie::vector<int32,VECTOR_SIZE> first = *inIt++;
        aie::vector<int32,VECTOR_SIZE> last = *inIt2++;
        aie::vector<int32,VECTOR_SIZE> first_v = first;
        aie::vector<int32,VECTOR_SIZE> last_v = last;
            LOG_VECTOR(first_v, true, "first=");
            LOG_VECTOR(last_v, true, "last=");
        v8acc80 res = lmul8(concat(first, last), 0, 0x32103210, 8, cst, 0, 0x22220000, 1);
        v8acc80 res2 = lmul8(concat(first, last), 0, 0x76547654, 8, cst, 0, 0x22220000, 1);

        aie::accum<acc80,VECTOR_SIZE> res_v = res;
        aie::accum<acc80,VECTOR_SIZE> res2_v = res2;
            LOG_VECTOR(res_v.template to_vector_zip<int32>(0), true, "res as vec=");
            LOG_VECTOR(res2_v.template to_vector_zip<int32>(0), true, "res2 as vec=");

        *outIt++ = res_v.template to_vector_zip<int32>(0);
        *outIt++ = res2_v.template to_vector_zip<int32>(0);


        // second iteration
        aie::vector<int32,VECTOR_SIZE> first_d = *inIt++;
        aie::vector<int32,VECTOR_SIZE> last_d = *inIt2++;
        aie::vector<int32,VECTOR_SIZE> first_v_d = first_d;
        aie::vector<int32,VECTOR_SIZE> last_v_d = last_d;
            LOG_VECTOR(first_v_d, true, "first=");
            LOG_VECTOR(last_v_d, true, "last=");
        v8acc80 res_d = lmul8(concat(first_d, last_d), 0, 0x32103210, 8, cst, 0, 0x22220000, 1);
        v8acc80 res2_d = lmul8(concat(first_d, last_d), 0, 0x76547654, 8, cst, 0, 0x22220000, 1);

        aie::accum<acc80,VECTOR_SIZE> res_v_d = res_d;
        aie::accum<acc80,VECTOR_SIZE> res2_v_d = res2_d;
            LOG_VECTOR(res_v_d.template to_vector_zip<int32>(0), true, "res as vec=");
            LOG_VECTOR(res2_v_d.template to_vector_zip<int32>(0), true, "res2 as vec=");

        *outIt++ = res_v_d.template to_vector_zip<int32>(0);
        *outIt++ = res2_v_d.template to_vector_zip<int32>(0);
    }

}



template<unsigned LENGTH, unsigned VECTOR_SIZE>
inline void ntt_int32_GS_unshuffle_add_sub(int32* const input, int32* const add_out, int32* sub_out) {

    int32 __aie_dm_resource_a* __restrict in_buffer_start = (int32 __aie_dm_resource_a* __restrict) input;    
    auto inIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_start);
    
    int32 __aie_dm_resource_b* __restrict out_buffer_start = (int32 __aie_dm_resource_b*) sub_out;
    auto sub_outIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_start);

    int32 __aie_dm_resource_d* __restrict temp_buffer_start = (int32 __aie_dm_resource_d* __restrict) add_out;
    auto add_outIt = aie::begin_vector<VECTOR_SIZE>(temp_buffer_start);

    // 2cc per 32 samples
    // The loop is unrolled manually over two iterations. The operands are stored alternating in wr and wd such that operands can be read in while the computations are happening. This is (only) necessary if the operands come from the same bank 
    alignas(16) const int16 cst_arr[16] = {1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   
    aie::vector<int16,2*VECTOR_SIZE> cst = aie::load_v<2*VECTOR_SIZE>(cst_arr);
    for (unsigned j=0; j < LENGTH / (4 * VECTOR_SIZE); j++) {
        // First iteration
        // Read operands
        v8int32 chess_storage(wr0) first = *inIt++;
        v8int32 chess_storage(wr1) last = *inIt++;
        aie::vector<int32,VECTOR_SIZE> first_v = first;
        aie::vector<int32,VECTOR_SIZE> last_v = last;
            LOG_VECTOR(first_v, true, "first=");
            LOG_VECTOR(last_v, true, "last=");
        // add, sub and unshuffle
        v8acc80 add_v = lmul8(concat(first, last), 0, 0xeca86420, 1, cst, 0, 0x00000000, 1);
        v8acc80 sub_v = lmul8(concat(first, last), 0, 0xeca86420, 1, cst, 0, 0x22222222, 1);

        // store results
        aie::accum<acc80,VECTOR_SIZE> add = add_v;
        aie::accum<acc80,VECTOR_SIZE> sub = sub_v;
            LOG_VECTOR(add.template to_vector<int32>(0), true, "add=");
            LOG_VECTOR(sub.template to_vector<int32>(0), true, "sub=");
        *add_outIt++ = add.template to_vector<int32>(0);
        *sub_outIt++ = sub.template to_vector<int32>(0);



        // Second iteration
        // Read operands
        v8int32 chess_storage(wd0) first_d = *inIt++;
        v8int32 chess_storage(wd1) last_d = *inIt++;
        aie::vector<int32,VECTOR_SIZE> first_v_d = first_d;
        aie::vector<int32,VECTOR_SIZE> last_v_d = last_d;
            LOG_VECTOR(first_v_d, true, "first=");
            LOG_VECTOR(last_v_d, true, "last=");
        // add, sub and unshuffle
        v8acc80 add_v_d = lmul8(concat(first_d, last_d), 0, 0xeca86420, 1, cst, 0, 0x00000000, 1);
        v8acc80 sub_v_d = lmul8(concat(first_d, last_d), 0, 0xeca86420, 1, cst, 0, 0x22222222, 1);

        // store results
        aie::accum<acc80,VECTOR_SIZE> add_d = add_v_d;
        aie::accum<acc80,VECTOR_SIZE> sub_d = sub_v_d;
            LOG_VECTOR(add_d.template to_vector<int32>(0), true, "add=");
            LOG_VECTOR(sub_d.template to_vector<int32>(0), true, "sub=");
        *add_outIt++ = add_d.template to_vector<int32>(0);
        *sub_outIt++ = sub_d.template to_vector<int32>(0);

    }
}




template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
NTT_INT32_CT_NO_TO_BO_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE>::NTT_INT32_CT_NO_TO_BO_KERNEL_CLASS(int32 (&factors_)[LENGTH], int32 (&temp_buffer_)[LENGTH>>1]) : factors(factors_), temp_buffer(temp_buffer_) {}



template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
void NTT_INT32_CT_NO_TO_BO_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE>::NTT_CT_NO_to_BO(
        adf::input_buffer<int32> & __restrict in_data, 
        adf::output_buffer<int32> & __restrict out_data) {


    constexpr unsigned K_BARRETT = 32;

    int32* const in_data_start = in_data.data();
    int32* const in_data_middle = in_data.data() + (LENGTH>>1);
    int32* const out_data_start = out_data.data();
    int32* const out_data_middle = out_data.data() + (LENGTH>>1);
    int32* const twiddle_factors = factors;
    int32* const barrett_multipliers = factors + (LENGTH>>1);


    ntt_int32_constant_barrett_multiplier_multiplication<LENGTH/2, MODULUS, K_BARRETT, VECTOR_SIZE, LENGTH/2>(in_data_start, out_data_start);

    ntt_int32_barrett_multiplier_multiplication<LENGTH/2, MODULUS, K_BARRETT, VECTOR_SIZE, LENGTH/2>(in_data_middle, barrett_multipliers, out_data_middle);


    ntt_int32_constant_twiddle_multiplication_and_reduction<LENGTH/2, MODULUS, K_BARRETT, VECTOR_SIZE, LENGTH/2>(in_data_start, out_data_start, temp_buffer);
        
    ntt_int32_twiddle_multiplication_and_reduction<LENGTH/2, MODULUS, K_BARRETT, VECTOR_SIZE, LENGTH/2>(in_data_middle, twiddle_factors, out_data_middle, in_data_middle);
        


    ntt_int32_CT_add_sub_interleave<LENGTH, VECTOR_SIZE>(temp_buffer, in_data_middle, out_data_start);

}



template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
NTT_INT32_GS_BO_TO_NO_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE>::NTT_INT32_GS_BO_TO_NO_KERNEL_CLASS(int32 (&factors_)[LENGTH], int32 (&temp_buffer_)[LENGTH>>1]) : factors(factors_), temp_buffer(temp_buffer_) {}



template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
void NTT_INT32_GS_BO_TO_NO_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE>::NTT_GS_BO_to_NO(
        adf::input_buffer<int32> & __restrict in_data, 
        adf::output_buffer<int32> & __restrict out_data) {


    constexpr unsigned K_BARRETT = 32;

    int32* const in_data_start = in_data.data();
    int32* const in_data_middle = in_data.data() + (LENGTH>>1);
    int32* const out_data_start = out_data.data();
    int32* const out_data_middle = out_data.data() + (LENGTH>>1);
    int32* const twiddle_factors = factors;
    int32* const barrett_multipliers = factors + (LENGTH>>1);


    ntt_int32_GS_unshuffle_add_sub<LENGTH, VECTOR_SIZE>(in_data_start, temp_buffer, out_data_middle);


    ntt_int32_constant_barrett_multiplier_multiplication<LENGTH/2, MODULUS, K_BARRETT, VECTOR_SIZE, LENGTH/2>(temp_buffer, in_data_start);

    ntt_int32_barrett_multiplier_multiplication<LENGTH/2, MODULUS, K_BARRETT, VECTOR_SIZE, LENGTH/2>(out_data_middle, barrett_multipliers, in_data_middle);



    ntt_int32_constant_twiddle_multiplication_and_reduction<LENGTH/2, MODULUS, K_BARRETT, VECTOR_SIZE, LENGTH/2>(temp_buffer, in_data_start, out_data_start);
        
    ntt_int32_twiddle_multiplication_and_reduction<LENGTH/2, MODULUS, K_BARRETT, VECTOR_SIZE, LENGTH/2>(out_data_middle, twiddle_factors, in_data_middle, out_data_middle);
        



}




// Set round and saturation mode (although these should already be the defaults)
void set_modes_ntt_layer(void) {
    aie::tile tile_obj=aie::tile::current();
    tile_obj.set_rounding(aie::rounding_mode::symmetric_zero);
    tile_obj.set_saturation(aie::saturation_mode::none);
}
