
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
 * Radix-4 int16 NTT layer kernel (Section 5.2 of the paper).
 *
 * One radix-4 layer = three blocks (Equation 6 rewritten as Equation 7):
 *   block 1: barrett_multiplication          -> A^1..A^4 = inputs times the twiddle factors (4 loops, one per quarter,
 *                                               writing the four inputs of every 4-NTT next to each other)
 *   block 2: barrett_multiplication_constant -> A^5, A^6 = omega_4 * A^2, omega_4 * A^4
 *   block 3: r4_matmul                       -> the two {-1,0,1} matrix products of Equation 7 (4x VMUL + 4x VMAC
 *                                               per 64 outputs), left unreduced (lazy reduction)
 */




template<unsigned LENGTH, unsigned MODULUS, bool ZIP_OUTPUT, unsigned VECTOR_SIZE>
inline void barrett_multiplication(int16* const input, int16* const twiddle_factors, int16* const barrett_factors, int16* const output) {
    LOG_STATUS("barrett_multiplication:\n");


    int16 __aie_dm_resource_a* __restrict in_buffer_middle = (int16 __aie_dm_resource_a*) input;
    auto inMiddleIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_middle);

    int16 __aie_dm_resource_a* __restrict in_buffer_middle2 = (int16 __aie_dm_resource_a*) input;
    auto inMiddleIt2 = aie::begin_vector<VECTOR_SIZE>(in_buffer_middle2);

    int16 __aie_dm_resource_b* __restrict out_buffer_middle = (int16 __aie_dm_resource_b*) output;
    auto outMiddleIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_middle);

    int16 __aie_dm_resource_c* __restrict factors_buffer_start = (int16 __aie_dm_resource_c*) twiddle_factors;
    auto factorsIt = aie::begin_vector<VECTOR_SIZE>(factors_buffer_start);

    int16 __aie_dm_resource_c* __restrict factors_buffer_middle = (int16 __aie_dm_resource_c*) barrett_factors;
    auto factorsMiddleIt = aie::begin_vector<VECTOR_SIZE>(factors_buffer_middle);

    // 1cc per iteration
    constexpr unsigned K_barrett = 16;
    v16int16 chess_storage(wc0) mod_cst = null_v16int16(); // we have to set to zero first, otherwise the compiler does not schedule efficiently for some reason
    aie::vector<int16, VECTOR_SIZE> mod_cst_v = aie::broadcast<int16, VECTOR_SIZE>(-int16(MODULUS));
    mod_cst = mod_cst_v;
    for (unsigned j=0; j < LENGTH / VECTOR_SIZE; j++) {

            LOG("j=: %u", j);
        
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


            aie::vector<int16, VECTOR_SIZE> lastTerms2_v = lastTerms2;
            LOG_VECTOR(lastTerms2_v, true, "\ninput2=");
            aie::vector<int16, VECTOR_SIZE> twiddle_v = twiddle;
            LOG_VECTOR(twiddle_v, true, "\ntwiddle=");
            LOG_VECTOR(mul_acc_v, true, "\nreduced acc=");


        if constexpr (ZIP_OUTPUT) {
            *outMiddleIt = mul_acc_v.template to_vector_zip<int16>(0); outMiddleIt += 4;
            LOG_VECTOR(mul_acc_v.template to_vector_zip<int16>(0), true, "\nreduced and zipped=");
        } else {
            *outMiddleIt = mul_acc_v.template to_vector<int16>(0); outMiddleIt += 4;
            LOG_VECTOR(mul_acc_v.template to_vector<int16>(0), true, "\nreduced=");
        }

    }
}

template<unsigned LENGTH, unsigned MODULUS, unsigned FACTOR, unsigned VECTOR_SIZE>
inline void barrett_multiplication_constant(int16* const input, int16* const output) {
    LOG_STATUS("barrett_multiplication_constant:\n");
    
    int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*) input + VECTOR_SIZE;
    auto inIt = aie::begin_vector<VECTOR_SIZE>(in_buffer_start);

    int16 __aie_dm_resource_a* __restrict in_buffer_start2 = (int16 __aie_dm_resource_a*) input + VECTOR_SIZE;
    auto inIt2 = aie::begin_vector<VECTOR_SIZE>(in_buffer_start2);

    int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*) output;
    auto outIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_start);

    // 1cc per iteration
    constexpr unsigned K_barrett = 16;
    v16int16 chess_storage(wc0) mod_cst = null_v16int16(); // we have to set to zero first, otherwise the compiler does not schedule efficiently for some reason
    const int16 factor_neg = FACTOR <= (MODULUS-1)/2 ? FACTOR : FACTOR - MODULUS; 
    const int16 barrett_factor = int16(double(factor_neg<<K_barrett) / double(MODULUS) + double(0.5));
    LOG("constant twiddle factor: %u", FACTOR);
    LOG("constant barrett factor: %d", barrett_factor);
    aie::vector<int16, VECTOR_SIZE> mod_cst_v = aie::concat(aie::broadcast<int16, VECTOR_SIZE/2>(int16(factor_neg)), aie::broadcast<int16, VECTOR_SIZE/2>(-int16(MODULUS)));
    aie::vector<int16, VECTOR_SIZE> barrett_cst_v = aie::broadcast<int16, VECTOR_SIZE>(int16(barrett_factor));
    mod_cst = mod_cst_v;
    for (unsigned j=0; j < LENGTH / VECTOR_SIZE; j++) {
        
        v16int16 lastTerms = *inIt; inIt += 2;
        v16int16 barrett = barrett_cst_v;
        v16acc48 mul_barrett_acc = mul16(concat(lastTerms, barrett), 0, 0x33221100, 0x77665544, 0x1100, 16, 0x33221100, 0x77665544, 0x1100);
        v16int16 mb_vec = srs(mul_barrett_acc, K_barrett+1);

            aie::vector<int16, VECTOR_SIZE> lastTerms_v = lastTerms;
            LOG_VECTOR(lastTerms_v, true, "\ninput=");
            aie::accum<acc48, VECTOR_SIZE> mul_barrett_acc_v = mul_barrett_acc;
            LOG_VECTOR(mul_barrett_acc_v, true, "\nmb_acc=");
            aie::vector<int16, VECTOR_SIZE> mb_vec_v = mb_vec;
            LOG_VECTOR(mb_vec_v, true, "\nmb_vec=");


        v16int16 lastTerms2 = *inIt2; inIt2 += 2;
        v16acc48 mul_acc = mul16(concat(lastTerms2, mb_vec), 0, 0x73727170, 0x77767574, 0x3120, concat(null_v16int16(), mod_cst), 16, 0x30303030, 0x30303030, 0x3120);
        aie::accum<acc48, VECTOR_SIZE> mul_acc_v = mul_acc;
        

            aie::vector<int16, VECTOR_SIZE> lastTerms2_v = lastTerms2;
            LOG_VECTOR(lastTerms2_v, true, "\ninput2=");
            LOG_VECTOR(mul_acc_v, true, "\nreduced acc=");


        *outIt++ = mul_acc_v.template to_vector<int16>(0);
            LOG_VECTOR(mul_acc_v.template to_vector<int16>(0), true, "\nreduced=");


    }
}



template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
inline void r4_matmul(int16* const input, int16* const input_omega_4, int16* const output) {
    LOG_STATUS("r4_matmul:\n");

    
    int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*) input_omega_4;
    auto F_omega_4_it = aie::begin_vector<VECTOR_SIZE>(in_buffer_start);

    int16 __aie_dm_resource_b* __restrict in_buffer_start6 = (int16 __aie_dm_resource_b*) input;
    auto F_it = aie::begin_vector<VECTOR_SIZE>(in_buffer_start6);

    int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*) output;
    auto outIt = aie::begin_vector<VECTOR_SIZE>(out_buffer_start);

    v16int16 chess_storage(wc0) cst = null_v16int16();
    alignas(16) const int16 cst_arr[16] = {1, 1, 1, -1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};   
    aie::vector<int16,VECTOR_SIZE> cst_v = aie::load_v<VECTOR_SIZE>(cst_arr);
    cst = cst_v;

    for (unsigned j=0; j < LENGTH / (4* VECTOR_SIZE); j++) chess_unroll_loop(1) {
        
        aie::vector<int16, VECTOR_SIZE> F_1 = *(F_it+1);
        aie::vector<int16, VECTOR_SIZE> F_3 = *(F_it+3);
        aie::vector<int16, VECTOR_SIZE> F_1_omega = *F_omega_4_it++;
        aie::vector<int16, VECTOR_SIZE> F_3_omega = *F_omega_4_it++;

            LOG_VECTOR(F_1, true, "F_1=");
            LOG_VECTOR(F_3, true, "F_3=");
            LOG_VECTOR(F_1_omega, true, "F_1_omega=");
            LOG_VECTOR(F_3_omega, true, "F_3_omega=");
        // requires data to be interleaved within each vector upon input!!
        // incorporate bitrev, this is actually useful in our computation!
        v16acc48 chess_storage(bm0) part_0 = mul16(concat(F_1, F_1_omega, F_3, F_3_omega), 0, 0xf9f1f8f0, 0xfbf3faf2, 0x2020, cst, 0, 0x42604260, 0x42604260, 1);
        v16acc48 chess_storage(bm1) part_1 = mul16(concat(F_1, F_1_omega, F_3, F_3_omega), 0, 0xfdf5fcf4, 0xfff7fef6, 0x2020, cst, 0, 0x42604260, 0x42604260, 1);
        v16acc48 chess_storage(bm2) part_2 = mul16(concat(F_1, F_1_omega, F_3, F_3_omega), 0, 0xf9f1f8f0, 0xfbf3faf2, 0x3131, cst, 0, 0x42604260, 0x42604260, 1);
        v16acc48 chess_storage(bm3) part_3 = mul16(concat(F_1, F_1_omega, F_3, F_3_omega), 0, 0xfdf5fcf4, 0xfff7fef6, 0x3131, cst, 0, 0x42604260, 0x42604260, 1);
    
        aie::accum<acc48, VECTOR_SIZE> part0 = part_0;
            LOG_VECTOR(part0.template to_vector<int16>(0), true, "part0=");
        aie::accum<acc48, VECTOR_SIZE> part1 = part_1;
            LOG_VECTOR(part1.template to_vector<int16>(0), true, "part1=");
        aie::accum<acc48, VECTOR_SIZE> part2 = part_2;
            LOG_VECTOR(part2.template to_vector<int16>(0), true, "part2=");
        aie::accum<acc48, VECTOR_SIZE> part3 = part_3;
            LOG_VECTOR(part3.template to_vector<int16>(0), true, "part3=");



        aie::vector<int16, VECTOR_SIZE> F_0 = *F_it;
        aie::vector<int16, VECTOR_SIZE> F_2 = *(F_it+2); F_it += 4;
            LOG_VECTOR(F_0, true, "F_0=");
            LOG_VECTOR(F_2, true, "F_2=");
        // remember: input data is interleaved within each vector!
        v16acc48 chess_storage(bm0) total_0 = mac16(part_0, concat(F_0,F_2), 0, 0x71717070, 0x73737272, 0x2020, cst, 0, 0x22002200, 0x22002200, 1);
        v16acc48 chess_storage(bm1) total_1 = mac16(part_1, concat(F_0,F_2), 0, 0x75757474, 0x77777676, 0x2020, cst, 0, 0x22002200, 0x22002200, 1);
        v16acc48 chess_storage(bm2) total_2 = mac16(part_2, concat(F_0,F_2), 0, 0x71717070, 0x73737272, 0x3131, cst, 0, 0x22002200, 0x22002200, 1);
        v16acc48 chess_storage(bm3) total_3 = mac16(part_3, concat(F_0,F_2), 0, 0x75757474, 0x77777676, 0x3131, cst, 0, 0x22002200, 0x22002200, 1);

        // store the vectors in a register before storing to memory, since VST.SRSS may cause memory stalls
        v16int16 chess_storage(WC) total_0_vec = srs(total_0, 0);
        *outIt++ = total_0_vec;
        v16int16 chess_storage(WC) total_1_vec = srs(total_1, 0);
        *outIt++ = total_1_vec;
        v16int16 chess_storage(WC) total_2_vec = srs(total_2, 0);
        *outIt++ = total_2_vec;
        v16int16 chess_storage(WC) total_3_vec = srs(total_3, 0);
        *outIt++ = total_3_vec;


        aie::accum<acc48, VECTOR_SIZE> total0 = total_0;
            LOG_VECTOR(total0.template to_vector<int16>(0), true, "total0=");
        aie::accum<acc48, VECTOR_SIZE> total1 = total_1;
            LOG_VECTOR(total1.template to_vector<int16>(0), true, "total1=");
        aie::accum<acc48, VECTOR_SIZE> total2 = total_2;
            LOG_VECTOR(total2.template to_vector<int16>(0), true, "total2=");
        aie::accum<acc48, VECTOR_SIZE> total3 = total_3;
            LOG_VECTOR(total3.template to_vector<int16>(0), true, "total3=");



    }

}


template<unsigned LENGTH, unsigned MODULUS, unsigned ROOT_4, unsigned VECTOR_SIZE>
NTT_R4_CT_NO_TO_BO_KERNEL_CLASS<LENGTH, MODULUS, ROOT_4, VECTOR_SIZE>::NTT_R4_CT_NO_TO_BO_KERNEL_CLASS(int16 (&factors_)[2*LENGTH], int16 (&temp_buffer_)[LENGTH>>1]) : factors(factors_), temp_buffer(temp_buffer_) {}



template<unsigned LENGTH, unsigned MODULUS, unsigned ROOT_4, unsigned VECTOR_SIZE>
void NTT_R4_CT_NO_TO_BO_KERNEL_CLASS<LENGTH, MODULUS, ROOT_4, VECTOR_SIZE>::NTT_CT_NO_to_BO(
        adf::input_buffer<int16> & __restrict in_data, 
        adf::output_buffer<int16> & __restrict out_data) {

        
    // perform in four different loops because the four inputs of an 4-NTT (per batch of 16) must be consecutive for the in-place matmul to function correctly
    // Manually unrolling the big loop over four iterations did not give an efficient pipeline. Also begin_pattern was not effective
    barrett_multiplication<LENGTH/4, MODULUS, true, VECTOR_SIZE>(in_data.data() + (0*LENGTH/4), factors + (0*LENGTH/4), factors + (4*LENGTH/4), out_data.data() + 0*VECTOR_SIZE);
    barrett_multiplication<LENGTH/4, MODULUS, true, VECTOR_SIZE>(in_data.data() + (1*LENGTH/4), factors + (1*LENGTH/4), factors + (5*LENGTH/4), out_data.data() + 1*VECTOR_SIZE);
    barrett_multiplication<LENGTH/4, MODULUS, true, VECTOR_SIZE>(in_data.data() + (2*LENGTH/4), factors + (2*LENGTH/4), factors + (6*LENGTH/4), out_data.data() + 2*VECTOR_SIZE);
    barrett_multiplication<LENGTH/4, MODULUS, true, VECTOR_SIZE>(in_data.data() + (3*LENGTH/4), factors + (3*LENGTH/4), factors + (7*LENGTH/4), out_data.data() + 3*VECTOR_SIZE);

    barrett_multiplication_constant<LENGTH/2, MODULUS, ROOT_4, VECTOR_SIZE>(out_data.data(), in_data.data());

    r4_matmul<LENGTH, MODULUS, VECTOR_SIZE>(out_data.data(), in_data.data(), out_data.data());


}








// Set round and saturation mode (although these should already be the defaults)
void set_modes_ntt_layer(void) {
    aie::tile tile_obj=aie::tile::current();
    tile_obj.set_rounding(aie::rounding_mode::symmetric_zero);
    tile_obj.set_saturation(aie::saturation_mode::none);
}


