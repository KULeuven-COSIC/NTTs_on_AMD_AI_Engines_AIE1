
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
 * Radix-2 int16 NTT layer kernels (Section 5.1 of the paper).
 *
 * Every radix-2 layer runs on its own AIE tile and is split (manual loop fission, Section 3) into
 * small software-pipelined loops, each implemented by one of the always-inlined helpers below:
 *   - barrett_reduction         : partial Barrett reduction (Section 4.2, Figure 3d without twiddle), II = 2 per 16 lanes
 *   - barrett_multiplication    : partial Barrett multiplication (Section 4.2, Figure 3d), II = 2 per 16 lanes
 *   - CT_add_sub_interleave     : butterfly add/sub fused with the perfect shuffle (block 3 of Section 5.1)
 *   - GS_unshuffle_add_sub,
 *     CT_add_sub_unshuffle,
 *     CT_add_sub                : butterflies fused with the unshuffle permutation for the inverse transform
 * The permutations are folded into the MAC data path through the lane-selection arguments of the
 * mul16 intrinsic, multiplying with a constant vector of +-1 entries (cst).
 *
 * Memory banks: __aie_dm_resource_{a,b,c,d} annotate which buffers are accessed concurrently, so
 * the compiler can avoid memory-bank conflicts (bank-disjointness invariant, Section 3.2).
 * chess_storage(...) pins variables to registers to obtain the intended schedule (Section 3.3, step 4).
 * The barrett_* helpers are called on chunks of LENGTH/4 coefficients so that all temporary buffers
 * fit in the data memory of one tile.
 */



// UNSHUFFLE_PAIR: indicates that the unshuffle must happen in pairs instead of single element. This is used in CT_BO_TO_NO
template<unsigned LENGTH, unsigned VECTOR_SIZE, bool UNSHUFFLE_PAIR>
inline __attribute__((always_inline)) void GS_unshuffle_add_sub(int16* const input, int16* const add_out, int16* const sub_out) { 

    int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a* __restrict) input;    
    auto inIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_start);
    
    int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*) sub_out;
    auto outIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_start);

    int16 __aie_dm_resource_d* __restrict temp_buffer_start = (int16 __aie_dm_resource_d* __restrict) add_out;
    auto tempIt = aie::begin_vector<VECTOR_SIZE>(temp_buffer_start);

    // 2cc per 32 samples
    // The loop is unrolled manually over two iterations. The operands are stored alternating in wr and wd such that operands can be read in while the computations are happening. This is (only) necessary if the operands come from the same bank 
    alignas(16) const int16 cst_arr[16] = {1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   
    aie::vector<int16,VECTOR_SIZE> cst = aie::load_v<VECTOR_SIZE>(cst_arr);
    for (unsigned j=0; j < LENGTH / (4 * VECTOR_SIZE); j++) {
        // First iteration
        // Read operands
        v16int16 chess_storage(wr0) first = *inIt++;
        v16int16 chess_storage(wr1) last = *inIt++;
        aie::vector<int16,VECTOR_SIZE> first_v = first;
        aie::vector<int16,VECTOR_SIZE> last_v = last;
            LOG_VECTOR(first_v, true, "first=");
            LOG_VECTOR(last_v, true, "last=");
        // add, sub and unshuffle
        v16acc48 add_v = mul16(concat(first, last), 0, 0x06040200, 0x0e0c0a08, UNSHUFFLE_PAIR ? 0x1010 : 0x3210, cst, 0, UNSHUFFLE_PAIR ? 0x20202020 : 0x00000000, UNSHUFFLE_PAIR ? 0x20202020 : 0x00000000, 1);
        v16acc48 sub_v = mul16(concat(first, last), 0, UNSHUFFLE_PAIR ? 0x07050301 : 0x06040200, UNSHUFFLE_PAIR ? 0x0f0d0b09 : 0x0e0c0a08, UNSHUFFLE_PAIR ? 0x1010 : 0x3210, cst, 0, UNSHUFFLE_PAIR ? 0x20202020 : 0x22222222, UNSHUFFLE_PAIR ? 0x20202020 : 0x22222222, 1);

        // store results
        aie::accum<acc48,VECTOR_SIZE> add = add_v;
        aie::accum<acc48,VECTOR_SIZE> sub = sub_v;
            LOG_VECTOR(add.template to_vector<int16>(0), true, "add=");
            LOG_VECTOR(sub.template to_vector<int16>(0), true, "sub=");
        *tempIt++ = add.template to_vector<int16>(0);
        *outIt++ = sub.template to_vector<int16>(0);



        // Second iteration
        // Read operands
        v16int16 chess_storage(wd0) first_d = *inIt++;
        v16int16 chess_storage(wd1) last_d = *inIt++;
        aie::vector<int16,VECTOR_SIZE> first_v_d = first_d;
        aie::vector<int16,VECTOR_SIZE> last_v_d = last_d;
            LOG_VECTOR(first_v_d, true, "first=");
            LOG_VECTOR(last_v_d, true, "last=");
        // add, sub and unshuffle
        v16acc48 add_v_d = mul16(concat(first_d, last_d), 0, 0x06040200, 0x0e0c0a08, UNSHUFFLE_PAIR ? 0x1010 : 0x3210, cst, 0, UNSHUFFLE_PAIR ? 0x20202020 : 0x00000000, UNSHUFFLE_PAIR ? 0x20202020 : 0x00000000, 1);
        v16acc48 sub_v_d = mul16(concat(first_d, last_d), 0, UNSHUFFLE_PAIR ? 0x07050301 : 0x06040200, UNSHUFFLE_PAIR ? 0x0f0d0b09 : 0x0e0c0a08, UNSHUFFLE_PAIR ? 0x1010 : 0x3210, cst, 0, UNSHUFFLE_PAIR ? 0x20202020 : 0x22222222, UNSHUFFLE_PAIR ? 0x20202020 : 0x22222222, 1);

        // store results
        aie::accum<acc48,VECTOR_SIZE> add_d = add_v_d;
        aie::accum<acc48,VECTOR_SIZE> sub_d = sub_v_d;
            LOG_VECTOR(add_d.template to_vector<int16>(0), true, "add=");
            LOG_VECTOR(sub_d.template to_vector<int16>(0), true, "sub=");
        *tempIt++ = add_d.template to_vector<int16>(0);
        *outIt++ = sub_d.template to_vector<int16>(0);

    }

}

template<unsigned LENGTH, unsigned VECTOR_SIZE>
inline __attribute__((always_inline)) void CT_add_sub_interleave(int16* const input, int16* const output) {

    int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*) input;
    auto inIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_start);

    int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*) output;
    auto outIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_start);

    int16 __aie_dm_resource_a* __restrict temp_buffer_start = (int16 __aie_dm_resource_a*) input + (LENGTH/2);
    auto tempIt = aie::begin_vector<VECTOR_SIZE>(temp_buffer_start);

    // 2cc per iteration 
    // The loop is unrolled manually over two iterations. The operands are stored alternating in wr and wd such that operands can be read in while the computations are happening. This is (only) necessary if the operands come from the same bank 
    alignas(16) const int16 cst_arr[16] = {1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   
    aie::vector<int16,VECTOR_SIZE> cst = aie::load_v<VECTOR_SIZE>(cst_arr); 
    for (unsigned j=0; j < LENGTH / (4 * VECTOR_SIZE); j++) {
            aie::vector<int16, VECTOR_SIZE> add = *inIt++;
            aie::vector<int16, VECTOR_SIZE> sub = *tempIt++; // ok to load both from input bank, needs at least two cc per iteration to write to output bank
                LOG_VECTOR(add, true, "\nadd=");
                LOG_VECTOR(sub, true, "\nsub=");
        v16acc48 res = mul16(concat(add, sub), 0, 0x73727170, 0x73727170, 0x3120, cst, 0, 0x00000000, 0x22222222, 1);
        v16acc48 res2 = mul16(concat(add, sub), 0, 0x77767574, 0x77767574, 0x3120, cst, 0, 0x00000000, 0x22222222, 1);

        aie::accum<acc48,VECTOR_SIZE> res_v = res;
        aie::accum<acc48,VECTOR_SIZE> res2_v = res2;
            LOG_VECTOR(res_v.template to_vector_zip<int16>(0), true, "res as vec=");
            LOG_VECTOR(res2_v.template to_vector_zip<int16>(0), true, "res2 as vec=");

        *outIt++ = res_v.template to_vector_zip<int16>(0);
        *outIt++ = res2_v.template to_vector_zip<int16>(0);


        // second iteration
        aie::vector<int16, VECTOR_SIZE> add_d = *inIt++;
        aie::vector<int16, VECTOR_SIZE> sub_d = *tempIt++; // ok to load both from input bank, needs at least two cc per iteration to write to output bank
        v16acc48 res_d = mul16(concat(add_d, sub_d), 0, 0x73727170, 0x73727170, 0x3120, cst, 0, 0x00000000, 0x22222222, 1);
        v16acc48 res2_d = mul16(concat(add_d, sub_d), 0, 0x77767574, 0x77767574, 0x3120, cst, 0, 0x00000000, 0x22222222, 1);

        aie::accum<acc48,VECTOR_SIZE> res_v_d = res_d;
        aie::accum<acc48,VECTOR_SIZE> res2_v_d = res2_d;
            LOG_VECTOR(res_v_d.template to_vector_zip<int16>(0), true, "res as vec=");
            LOG_VECTOR(res2_v_d.template to_vector_zip<int16>(0), true, "res2 as vec=");

        *outIt++ = res_v_d.template to_vector_zip<int16>(0);
        *outIt++ = res2_v_d.template to_vector_zip<int16>(0);

    }
}


template<unsigned LENGTH, unsigned VECTOR_SIZE>
inline __attribute__((always_inline)) void CT_add_sub_unshuffle(int16* const input_low, int16* const input_high, int16* const out_Q0, int16* const out_Q1, int16* const out_Q2, int16* const out_Q3) { 

    int16 __aie_dm_resource_a* __restrict in_buffer_low_start = (int16 __aie_dm_resource_a* __restrict) input_low;    
    auto inLow0It = aie::begin_vector<VECTOR_SIZE>(in_buffer_low_start);

    int16 __aie_dm_resource_a* __restrict in_buffer_low_start2 = (int16 __aie_dm_resource_a* __restrict) input_low + VECTOR_SIZE;    
    auto inLow1It = aie::begin_vector<VECTOR_SIZE>(in_buffer_low_start2);
    
    int16 __aie_dm_resource_b* __restrict in_buffer_high_start = (int16 __aie_dm_resource_b*) input_high;
    auto inHighIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_high_start);

    int16 __aie_dm_resource_d* __restrict out_buffer_low_start = (int16 __aie_dm_resource_d* __restrict) out_Q0;
    auto outQ0It = aie::begin_vector<VECTOR_SIZE>(out_buffer_low_start);

    int16 __aie_dm_resource_d* __restrict out_buffer_q2_start = (int16 __aie_dm_resource_d* __restrict) out_Q1;
    auto outQ1It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q2_start);

    int16 __aie_dm_resource_d* __restrict out_buffer_q3_start = (int16 __aie_dm_resource_d* __restrict) out_Q2;
    auto outQ2It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q3_start);

    int16 __aie_dm_resource_d* __restrict out_buffer_q4_start = (int16 __aie_dm_resource_d* __restrict) out_Q3;
    auto outQ3It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q4_start);

   
   // 4cc per 64 samples = 2cc per 32 butterflies
   alignas(16) const int16 cst_arr[16] = {1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   
    aie::vector<int16,VECTOR_SIZE> cst = aie::load_v<VECTOR_SIZE>(cst_arr);
    v16int16 chess_storage(wc0) cst_v = cst;
    // The loop is manually unrolled over two iterations. This is because we need two cycles to read in the four vectors. 
    // Hence, we must use different set of registers in subsequent iterations such that we can 
    // already load in two vectors of the next iteration while the previous iteration is still starting computations 
    // (so their registers cannot be overwritten yet)
       for (unsigned j=0; j < LENGTH / (8 * VECTOR_SIZE); j++) {
        // First iteration
        {
        // Read operands
        v16int16 chess_storage(wr0) high_0 = *inHighIt++;
        v16int16 chess_storage(wr1) low_1 = *inLow1It; inLow1It += 2;
        v16int16 chess_storage(wr2) low_0 = *inLow0It; inLow0It += 2;
        v16int16 chess_storage(wr3) high_1 = *inHighIt++;
        
        // add, sub and unshuffle
        v16acc48 chess_storage(bm0) add1_v = mul16(concat(high_0, low_1, low_0, high_1), 16, 0xfefcfaf8, 0xf6f4f2f0, 0x3120, cst_v, 0, 0x00000000, 0x00000000, 1);
        v16acc48 chess_storage(bm0) add2_v = mul16(concat(high_0, low_1, low_0, high_1), 16, 0xfffdfbf9, 0xf7f5f3f1, 0x3120, cst_v, 0, 0x00000000, 0x00000000, 1);
        v16acc48 chess_storage(bm0) sub1_v = mul16(concat(high_0, low_1, low_0, high_1), 16, 0xfefcfaf8, 0xf6f4f2f0, 0x3120, cst_v, 0, 0x22222222, 0x22222222, 1);
        v16acc48 chess_storage(bm0) sub2_v = mul16(concat(high_0, low_1, low_0, high_1), 16, 0xfffdfbf9, 0xf7f5f3f1, 0x3120, cst_v, 0, 0x22222222, 0x22222222, 1);

        // store results
        *outQ0It++ = srs(add1_v,0);
        *outQ2It++ = srs(add2_v,0);
        *outQ1It++ = srs(sub1_v,0);
        *outQ3It++ = srs(sub2_v,0);


        aie::vector<int16, 16> low_0_v = low_0; LOG_VECTOR(low_0_v, true, "low_0=");
        aie::vector<int16, 16> low_1_v = low_1; LOG_VECTOR(low_1_v, true, "low_1=");
        aie::vector<int16, 16> high_0_v = high_0; LOG_VECTOR(high_0_v, true, "high_0=");
        aie::vector<int16, 16> high_1_v = high_1; LOG_VECTOR(high_1_v, true, "high_1=");
        aie::vector<int16, 16> out0 = srs(add1_v,0); LOG_VECTOR(out0, true, "out0=");
        aie::vector<int16, 16> out1 = srs(sub1_v,0); LOG_VECTOR(out1, true, "out1=");
        aie::vector<int16, 16> out2 = srs(add2_v,0); LOG_VECTOR(out2, true, "out2=");
        aie::vector<int16, 16> out3 = srs(sub2_v,0); LOG_VECTOR(out3, true, "out3=");
        }

        // Second iteration
        {
        // Read operands
        v16int16 chess_storage(wd0) high_0 = *inHighIt++;
        v16int16 chess_storage(wd1) low_1 = *inLow1It; inLow1It += 2;
        v16int16 chess_storage(wr2) low_0 = *inLow0It; inLow0It += 2;
        v16int16 chess_storage(wr3) high_1 = *inHighIt++;
        
        // add, sub and unshuffle
        v16acc48 chess_storage(bm0) add1_v = mul16(concat(high_0, low_1, low_0, high_1), 16, 0xfefcfaf8, 0xf6f4f2f0, 0x3120, cst_v, 0, 0x00000000, 0x00000000, 1);
        v16acc48 chess_storage(bm0) add2_v = mul16(concat(high_0, low_1, low_0, high_1), 16, 0xfffdfbf9, 0xf7f5f3f1, 0x3120, cst_v, 0, 0x00000000, 0x00000000, 1);
        v16acc48 chess_storage(bm0) sub1_v = mul16(concat(high_0, low_1, low_0, high_1), 16, 0xfefcfaf8, 0xf6f4f2f0, 0x3120, cst_v, 0, 0x22222222, 0x22222222, 1);
        v16acc48 chess_storage(bm0) sub2_v = mul16(concat(high_0, low_1, low_0, high_1), 16, 0xfffdfbf9, 0xf7f5f3f1, 0x3120, cst_v, 0, 0x22222222, 0x22222222, 1);

        // store results
        *outQ0It++ = srs(add1_v,0);
        *outQ2It++ = srs(add2_v,0);
        *outQ1It++ = srs(sub1_v,0);
        *outQ3It++ = srs(sub2_v,0);



        aie::vector<int16, 16> low_0_v = low_0; LOG_VECTOR(low_0_v, true, "low_0=");
        aie::vector<int16, 16> low_1_v = low_1; LOG_VECTOR(low_1_v, true, "low_1=");
        aie::vector<int16, 16> high_0_v = high_0; LOG_VECTOR(high_0_v, true, "high_0=");
        aie::vector<int16, 16> high_1_v = high_1; LOG_VECTOR(high_1_v, true, "high_1=");
        aie::vector<int16, 16> out0 = srs(add1_v,0); LOG_VECTOR(out0, true, "out0=");
        aie::vector<int16, 16> out1 = srs(sub1_v,0); LOG_VECTOR(out1, true, "out1=");
        aie::vector<int16, 16> out2 = srs(add2_v,0); LOG_VECTOR(out2, true, "out2=");
        aie::vector<int16, 16> out3 = srs(sub2_v,0); LOG_VECTOR(out3, true, "out3=");
        }
    }


}


template<unsigned LENGTH, unsigned VECTOR_SIZE>
inline __attribute__((always_inline)) void CT_add_sub(int16* const input_low, int16* const input_high, int16* const out_add, int16* const out_sub) { 

    int16 __aie_dm_resource_a* __restrict in_buffer_low_start = (int16 __aie_dm_resource_a* __restrict) input_low;    
    auto inLowIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_low_start);

    int16 __aie_dm_resource_b* __restrict in_buffer_high_start = (int16 __aie_dm_resource_b*) input_high;
    auto inHighIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_high_start);

    int16 __aie_dm_resource_d* __restrict out_buffer_low_start = (int16 __aie_dm_resource_d* __restrict) out_add;
    auto outAddIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_low_start);

    int16 __aie_dm_resource_d* __restrict out_buffer_q2_start = (int16 __aie_dm_resource_d* __restrict) out_sub;
    auto outSubIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_q2_start);

   
   // 4cc per 64 samples = 2cc per 32 butterflies
   alignas(16) const int16 cst_arr[16] = {1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   
    aie::vector<int16,VECTOR_SIZE> cst = aie::load_v<VECTOR_SIZE>(cst_arr);
    v16int16 chess_storage(wc0) cst_v = cst;
       for (unsigned j=0; j < LENGTH / (2 * VECTOR_SIZE); j++) {
        v16int16 chess_storage(wr1) high_0 = *inHighIt++;
        v16int16 chess_storage(wr0) low_0 = *inLowIt++;
        
        // add, sub and unshuffle
        v16acc48 chess_storage(bm0) add_v = mul16(concat(low_0, high_0), 0, 0x73727170, 0x77767574, 0x3120, cst_v, 0, 0x00000000, 0x00000000, 1);
        v16acc48 chess_storage(bm0) sub_v = mul16(concat(low_0, high_0), 0, 0x73727170, 0x77767574, 0x3120, cst_v, 0, 0x22222222, 0x22222222, 1);
       
        // store results
        *outAddIt++ = srs(add_v,0);
        *outSubIt++ = srs(sub_v,0);


        aie::vector<int16, 16> low_0_v = low_0; LOG_VECTOR(low_0_v, true, "low_0=");
        aie::vector<int16, 16> high_0_v = high_0; LOG_VECTOR(high_0_v, true, "high_0=");
        aie::vector<int16, 16> out0 = srs(add_v,0); LOG_VECTOR(out0, true, "add=");
        aie::vector<int16, 16> out1 = srs(sub_v,0); LOG_VECTOR(out1, true, "sub=");
    }

    

}





// Partial Barrett multiplication (Definition 1 with K = 16 and nearest-integer rounding):
//   output[i] = input[i] * twiddle[i] - round(input[i] * barrett[i] / 2^16) * q, with |output[i]| <= 3q/4
// The first mul16 computes a*w' + a*w' = 2*a*w' (mul16 always sums two products per lane, see
// Figure 3d), so the SRS path shifts by K+1 instead of K. The second mul16 computes a*w - m'*q in a
// single instruction by concatenating (a, m') with (w, -q).
template<unsigned LENGTH, unsigned MODULUS, bool FACTORS_SPLIT, unsigned VECTOR_SIZE>
inline __attribute__((always_inline)) void barrett_multiplication(int16* const input, int16* const twiddle_factors, int16* const barrett_factors, int16* output) {
    // boolean FACTORS_SPLIT indicates whether the twiddle and barrett factors are in the same memory bank or not
    // If not, it is assumed that the twiddle factors are in the same bank as the output
    // this has no impact on the schedule, only on the resourcec annotations

    int16 __aie_dm_resource_a* __restrict in_buffer_middle = (int16 __aie_dm_resource_a*) input;
    auto inMiddleIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_middle);

    int16 __aie_dm_resource_a* __restrict in_buffer_middle2 = (int16 __aie_dm_resource_a*) input;
    auto inMiddleIt2 = aie::begin_vector<VECTOR_SIZE>(in_buffer_middle2);

    int16 __aie_dm_resource_b* __restrict out_buffer_middle = (int16 __aie_dm_resource_b*) output;
    auto outMiddleIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_middle);

    int16* factors_buffer_start;
    if constexpr (FACTORS_SPLIT) {
        factors_buffer_start = (int16 __aie_dm_resource_b*) twiddle_factors;
    } else {
        factors_buffer_start = (int16 __aie_dm_resource_c*) twiddle_factors;
    }
    auto factorsIt = aie::begin_vector<VECTOR_SIZE>(factors_buffer_start);

    int16 __aie_dm_resource_c* __restrict factors_buffer_middle = (int16 __aie_dm_resource_c*) barrett_factors;
    auto factorsMiddleIt = aie::begin_vector<VECTOR_SIZE>(factors_buffer_middle);

    // 2cc per iteration (two MAC and two SRS passes, Figure 3d)
    constexpr unsigned K_barrett = 16;
    v16int16 chess_storage(wc0) mod_cst = null_v16int16(); // we have to set to zero first, otherwise the compiler does not schedule efficiently for some reason
    aie::vector<int16, VECTOR_SIZE> mod_cst_v = aie::broadcast<int16, VECTOR_SIZE>(-int16(MODULUS));
    mod_cst = mod_cst_v;
    for (unsigned j=0; j < LENGTH / VECTOR_SIZE; j++) {
        
        v16int16 lastTerms = *inMiddleIt++;
        v16int16 barrett = *factorsMiddleIt++;
        v16acc48 mul_barrett_acc = mul16(concat(lastTerms, barrett), 0, 0x33221100, 0x77665544, 0x1100, 16, 0x33221100, 0x77665544, 0x1100);
        v16int16 mb_vec = srs(mul_barrett_acc, K_barrett+1);

            aie::vector<int16, VECTOR_SIZE> lastTerms_v = lastTerms;
            LOG_VECTOR(lastTerms_v, true, "\ninput=");
            aie::vector<int16, VECTOR_SIZE> barrett_v = barrett;
            LOG_VECTOR(barrett_v, true, "\nbarrett=");
            aie::accum<acc48, VECTOR_SIZE> mul_barrett_acc_v = mul_barrett_acc;
            LOG_VECTOR(mul_barrett_acc_v, true, "\nmb_acc=");
            aie::vector<int16, VECTOR_SIZE> mb_vec_v = mb_vec;
            LOG_VECTOR(mb_vec_v, true, "\nmb_vec=");


        v16int16 lastTerms2 = *inMiddleIt2++;
        v16int16 twiddle = *factorsIt++; 
        v16acc48 mul_acc = mul16(concat(lastTerms2, mb_vec), 0, 0x73727170, 0x77767574, 0x3120, concat(twiddle, mod_cst), 0, 0x73727170, 0x77767574, 0x3120);
        aie::accum<acc48, VECTOR_SIZE> mul_acc_v = mul_acc;
        *outMiddleIt++ = mul_acc_v.template to_vector<int16>(0);


            aie::vector<int16, VECTOR_SIZE> lastTerms2_v = lastTerms2;
            LOG_VECTOR(lastTerms2_v, true, "\ninput2=");
            aie::vector<int16, VECTOR_SIZE> twiddle_v = twiddle;
            LOG_VECTOR(twiddle_v, true, "\ntwiddle=");
            LOG_VECTOR(mul_acc_v, true, "\nreduced acc=");
            LOG_VECTOR(mul_acc_v.template to_vector<int16>(0), true, "\nreduced=");
    }
}

// Partial Barrett reduction (Definition 1 with w = 1): same structure as barrett_multiplication, but
// the Barrett multiplier round(2^16/q) is a constant and no twiddle factors are loaded.
template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
inline __attribute__((always_inline)) void barrett_reduction(int16* const input, int16* output) {
    LOG_STATUS("barrett_reduction:\n");
    
    int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*) input;
    auto inIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_start);

    int16 __aie_dm_resource_a* __restrict in_buffer_start2 = (int16 __aie_dm_resource_a*) input;
    auto inIt2 = aie::begin_vector<VECTOR_SIZE>(in_buffer_start2);

    int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*) output;
    auto outIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_start);

    // 2cc per iteration (two MAC and two SRS passes, Figure 3d)
    constexpr unsigned K_barrett = 16;
    v16int16 chess_storage(wc0) mod_cst = null_v16int16(); // we have to set to zero first, otherwise the compiler does not schedule efficiently for some reason
    const int16 barrett_factor = int16(double(1<<K_barrett) / double(MODULUS) + double(0.5));
    LOG("barrett_factor: %d", barrett_factor);
    aie::vector<int16, VECTOR_SIZE> mod_cst_v = aie::concat(aie::broadcast<int16, VECTOR_SIZE/2>(int16(1)), aie::broadcast<int16, VECTOR_SIZE/2>(-int16(MODULUS)));
    aie::vector<int16, VECTOR_SIZE> barrett_cst_v = aie::broadcast<int16, VECTOR_SIZE>(int16(barrett_factor));
    mod_cst = mod_cst_v;
    for (unsigned j=0; j < LENGTH / VECTOR_SIZE; j++) {
        
        v16int16 lastTerms = *inIt++;
        v16int16 barrett = barrett_cst_v;
        v16acc48 mul_barrett_acc = mul16(concat(lastTerms, barrett), 0, 0x33221100, 0x77665544, 0x1100, 16, 0x33221100, 0x77665544, 0x1100);
        v16int16 mb_vec = srs(mul_barrett_acc, K_barrett+1);

            aie::vector<int16, VECTOR_SIZE> lastTerms_v = lastTerms;
            LOG_VECTOR(lastTerms_v, true, "\ninput=");
            aie::accum<acc48, VECTOR_SIZE> mul_barrett_acc_v = mul_barrett_acc;
            LOG_VECTOR(mul_barrett_acc_v, true, "\nmb_acc=");
            aie::vector<int16, VECTOR_SIZE> mb_vec_v = mb_vec;
            LOG_VECTOR(mb_vec_v, true, "\nmb_vec=");


        v16int16 lastTerms2 = *inIt2++;
        v16acc48 mul_acc = mul16(concat(lastTerms2, mb_vec), 0, 0x73727170, 0x77767574, 0x3120, concat(null_v16int16(), mod_cst), 16, 0x30303030, 0x30303030, 0x3120);
        aie::accum<acc48, VECTOR_SIZE> mul_acc_v = mul_acc;
        *outIt++ = mul_acc_v.template to_vector<int16>(0);


            aie::vector<int16, VECTOR_SIZE> lastTerms2_v = lastTerms2;
            LOG_VECTOR(lastTerms2_v, true, "\ninput2=");
            LOG_VECTOR(mul_acc_v, true, "\nreduced acc=");
            LOG_VECTOR(mul_acc_v.template to_vector<int16>(0), true, "\nreduced=");
    }
}




/* CT NO TO BO: forward radix-2 layer (Section 5.1, Figure 4b) */
// Each half of the polynomial is processed in three loops:
//   block 1: partial Barrett reduction of the top quarter,
//   block 2: partial Barrett multiplication of the bottom quarter with the twiddle factors,
//   block 3: butterflies (add/sub, left unreduced) fused with the perfect shuffle.
// Because of the perfect shuffle, every layer has the same structure (constant geometry), so all tiles run this kernel.
template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
NTT_CT_NO_TO_BO_IMPROVED_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE>::NTT_CT_NO_TO_BO_IMPROVED_KERNEL_CLASS(int16 (&factors_)[LENGTH], int16 (&temp_buffer_)[LENGTH>>1]) : factors(factors_), temp_buffer(temp_buffer_) {}

template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
void NTT_CT_NO_TO_BO_IMPROVED_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE>::NTT_CT_NO_to_BO(
        adf::input_buffer<int16> & __restrict in_data, 
        adf::output_buffer<int16> & __restrict out_data) {


    LOG_STATUS("NTT_CT_NO_to_BO: START\n");

    // barrett reduction of first part (FIRST HALF)
    LOG_STATUS("NTT_CT_NO_to_BO: barrett reduction of first part (FIRST HALF)\n");
    barrett_reduction<LENGTH/4, MODULUS, VECTOR_SIZE>(in_data.data(), temp_buffer);

    // barrett multiplication of second part (FIRST HALF)
    LOG_STATUS("NTT_CT_NO_to_BO: barrett reduction of second part (FIRST HALF)\n");
    barrett_multiplication<LENGTH/4, MODULUS, false, VECTOR_SIZE>(in_data.data() + (LENGTH/2), factors, factors + (LENGTH/2), temp_buffer + (LENGTH/4));

    // add/sub/ilv (FIRST HALF)
    LOG_STATUS("NTT_CT_NO_to_BO: add/sub/ilv (FIRST HALF)\n");
    CT_add_sub_interleave<LENGTH/2, VECTOR_SIZE>(temp_buffer, out_data.data());




    // barrett reduction of first part (SECOND HALF)
    LOG_STATUS("NTT_CT_NO_to_BO: barrett reduction of first part (SECOND HALF)\n");
    barrett_reduction<LENGTH/4, MODULUS, VECTOR_SIZE>(in_data.data() + (LENGTH/4), temp_buffer);

    // barrett multiplication of second part (SECOND HALF)
    LOG_STATUS("NTT_CT_NO_to_BO: barrett reduction of second part (SECOND HALF)\n");
    barrett_multiplication<LENGTH/4, MODULUS, false, VECTOR_SIZE>(in_data.data() + (3*LENGTH/4), factors + (LENGTH/4), factors + (3*LENGTH/4), temp_buffer + (LENGTH/4));

    // add/sub/ilv (SECOND HALF)
    LOG_STATUS("NTT_CT_NO_to_BO: add/sub/ilv (SECOND HALF)\n");
    CT_add_sub_interleave<LENGTH/2, VECTOR_SIZE>(temp_buffer, out_data.data() + (LENGTH/2));



   
    LOG_STATUS("CT_NO_to_BO: END\n");
}



/* BO TO NO: inverse radix-2 layer (Section 5.1, "Constraints and inverse transform") */
// Mirrors the forward kernel with an unshuffle permutation instead of a perfect shuffle.
// The first layer only needs add/sub + unshuffle (all its twiddle factors are 1); the last layer
// skips the unshuffle so that the output is in natural order.
template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE, unsigned NO_OF_LAYERS, unsigned LAYER>
NTT_CT_BO_TO_NO_IMPROVED_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE, NO_OF_LAYERS, LAYER>::NTT_CT_BO_TO_NO_IMPROVED_KERNEL_CLASS(int16 (&twiddle_factors_)[LENGTH/2], int16 (&barrett_multipliers_)[LENGTH/2], int16 (&temp_buffer_)[LENGTH/2], int16 (&temp_buffer2_)[LENGTH/4]) : twiddle_factors(twiddle_factors_), barrett_multipliers(barrett_multipliers_), temp_buffer(temp_buffer_), temp_buffer2(temp_buffer2_) {}

template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE, unsigned NO_OF_LAYERS, unsigned LAYER>
void NTT_CT_BO_TO_NO_IMPROVED_KERNEL_CLASS<LENGTH, MODULUS, VECTOR_SIZE, NO_OF_LAYERS, LAYER>::NTT_CT_BO_to_NO(
        adf::input_buffer<int16> & __restrict in_data,
        adf::output_buffer<int16> & __restrict out_data) {

    LOG_STATUS("CT_BO_to_NO: START\n");

    // Only Perform the first add/sub and an unshuffle of the single coefficients
    if constexpr (LAYER == 0) {
        // add, sub and unshuffle (SECOND HALF)
        LOG_STATUS("CT_BO_to_NO: (FIRST LAYER) add/sub/ilv\n");
        GS_unshuffle_add_sub<LENGTH, VECTOR_SIZE, true>(in_data.data(), out_data.data(), out_data.data() + LENGTH/2);

    } else {

        // Don't perform unshuffle in the last layer
        constexpr bool UNSHUFFLE = (LAYER != NO_OF_LAYERS-1); 
        LOG("UNSHUFFLE: %s\n", UNSHUFFLE ? "true" : "false");

        // barrett multiplication of sub with twiddle factor
        LOG_STATUS("CT_BO_to_NO: barrett multiplication of sub with twiddle factor\n");
        barrett_multiplication<LENGTH/2, MODULUS, true, VECTOR_SIZE>(in_data.data() + (LENGTH/2), twiddle_factors, barrett_multipliers, temp_buffer);




        // barrett reduction of add (FIRST HALF)
        LOG_STATUS("CT_BO_to_NO: barrett reduction of add (FIRST HALF)\n");
        barrett_reduction<LENGTH/4, MODULUS, VECTOR_SIZE>(in_data.data(), temp_buffer2);

        // add, sub and unshuffle (FIRST HALF)
        if constexpr (UNSHUFFLE) {
            LOG_STATUS("CT_BO_to_NO: add/sub/unshuffle (FIRST HALF)\n");
            CT_add_sub_unshuffle<LENGTH/2, VECTOR_SIZE>(temp_buffer2, temp_buffer, out_data.data(), out_data.data() + (2*LENGTH/8), out_data.data() + (4*LENGTH/8), out_data.data() + (6*LENGTH/8));
        } else {
            LOG_STATUS("CT_BO_to_NO: add/sub (FIRST HALF)\n");
            CT_add_sub<LENGTH/2, VECTOR_SIZE>(temp_buffer2, temp_buffer, out_data.data(), out_data.data() + (LENGTH/2));
        }



        // barrett reduction of add (SECOND HALF)
        LOG_STATUS("CT_BO_to_NO: barrett reduction of add (SECOND HALF)\n");
        barrett_reduction<LENGTH/4, MODULUS, VECTOR_SIZE>(in_data.data() + (LENGTH/4), temp_buffer2);

        // add, sub and unshuffle (SECOND HALF)
        if constexpr (UNSHUFFLE) {
            LOG_STATUS("CT_BO_to_NO: add/sub/unshuffle (SECOND HALF)\n");
            CT_add_sub_unshuffle<LENGTH/2, VECTOR_SIZE>(temp_buffer2, temp_buffer + (LENGTH/4), out_data.data() + (LENGTH/8), out_data.data() + (3*LENGTH/8), out_data.data() + (5*LENGTH/8), out_data.data() + (7*LENGTH/8));;
        } else {
            LOG_STATUS("CT_BO_to_NO: add/sub (FIRST HALF)\n");
            CT_add_sub<LENGTH/2, VECTOR_SIZE>(temp_buffer2, temp_buffer + (LENGTH/4), out_data.data() + (LENGTH/4), out_data.data() + (3*LENGTH/4));
        }
    }


    LOG_STATUS("CT_BO_to_NO: END\n");
}

// Set round and saturation mode (although these should already be the defaults)
void set_modes_ntt_layer(void) {
    aie::tile tile_obj=aie::tile::current();
    tile_obj.set_rounding(aie::rounding_mode::symmetric_zero);
    tile_obj.set_saturation(aie::saturation_mode::none);
}
