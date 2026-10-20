#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include "aie_api/types.hpp"
#include "debug.h"
#include "kernels.h"

#ifdef __X86SIM__
#include <adf/x86sim/x86simDebug.h>
#endif

#define BATCH_SIZE 16

/*
 * Full Barrett multiplication by a constant for int16 (Section 4.2, "Full reduction primitives").
 *
 * out[i] = MULTIPLICATION_FACTOR * in[i] mod MODULUS, fully reduced to [0, q). This can be used e.g.
 * for the final multiplication by n^{-1} of an inverse NTT combined with a full reduction.
 * Implemented in three loops (per half of the buffer for the last two):
 *   1. m'[i] = round(in[i] * w' / 2^16)                                     (Barrett multiplier)
 *   2. r[i]  = in[i] * w - m'[i] * q  (partial result in [-3q/4, 3q/4]) and its sign mask
 *      s[i] = floor(r[i] / 2^16) in {-1, 0}, extracted in the SRS path with floor rounding
 *   3. out[i] = r[i] - s[i] * q                                             (conditional addition of q)
 * Cost: 1 + 2 + 1 = 4 cycles per 16 coefficients (Table 1).
 */


template<unsigned LENGTH, unsigned MODULUS, unsigned MULTIPLICATION_FACTOR>
FULL_BARRETT_MULTIPLICATION_KERNEL_CLASS<LENGTH, MODULUS, MULTIPLICATION_FACTOR>::FULL_BARRETT_MULTIPLICATION_KERNEL_CLASS(int16 (&temp_buffer_)[LENGTH>>1], int16 (&temp_buffer2_)[LENGTH>>1]) : temp_buffer(temp_buffer_), temp_buffer2(temp_buffer2_) {}


template<unsigned LENGTH, unsigned MODULUS, unsigned MULTIPLICATION_FACTOR>
__attribute__((noinline)) void FULL_BARRETT_MULTIPLICATION_KERNEL_CLASS<LENGTH, MODULUS, MULTIPLICATION_FACTOR>::full_barrett_multiplication_kernel(
        adf::input_buffer<int16> & __restrict in_data,
        adf::output_buffer<int16> & __restrict out_data) {




    // Get the barret factor for 1
    aie::tile tile_obj=aie::tile::current(); //get the tile of the kernel
    tile_obj.set_rounding(aie::rounding_mode::symmetric_zero);
    constexpr uint16 K_barret = 16;
    const int16 factor_neg = MULTIPLICATION_FACTOR <= (MODULUS - 1) / 2 ? MULTIPLICATION_FACTOR : MULTIPLICATION_FACTOR - MODULUS;
    const int16 barret_factor = static_cast<int16>(std::round(float(int32(factor_neg)<<K_barret) / float(MODULUS) + 0.5f));

    LOG_STATUS("full Barrett multiplication");
    LOG("constant twiddle factor: %u", MULTIPLICATION_FACTOR);
    LOG("constant twiddle factor negative: %u", factor_neg);
    LOG("constant barrett factor: %u", barret_factor);

    // Mul with barrett factor
    {
        int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*)in_data.data();
        auto vectorINIt = aie::begin_vector<BATCH_SIZE>(in_buffer_start);
        int16 __aie_dm_resource_c* __restrict temp_buffer_start = (int16 __aie_dm_resource_c*) temp_buffer;
        auto tempIT = aie::begin_vector<BATCH_SIZE>(temp_buffer_start);
        int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*)out_data.data();
        auto vectorOUTIt = aie::begin_vector<BATCH_SIZE>(out_buffer_start);  

        // 1cc per iteration
        for (unsigned j=0; j < LENGTH / BATCH_SIZE; j++) { 
            aie::vector<int16, BATCH_SIZE> terms = *vectorINIt++;
                LOG_VECTOR(terms, true, "terms=");
            aie::accum<acc48, BATCH_SIZE> mul_barret = aie::mul(terms, barret_factor);
                LOG_VECTOR(mul_barret, true, "mul_barret=");
            aie::vector<int16, BATCH_SIZE> mul_barret_vec = mul_barret.to_vector<int16>(K_barret);
                LOG_VECTOR(mul_barret_vec, true, "mul_barret_vec=");
            *vectorOUTIt++ = mul_barret_vec;
        }
    }


    // Mul with twiddle factor + reduction to [-q,q] + determine sign
    // Only for first half of buffer, because temp_stack only half the size (because stack takes other half of bank)
    {
        int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*)in_data.data();
        auto InIt = aie::begin_vector<BATCH_SIZE>(in_buffer_start);
        int16 __aie_dm_resource_c* __restrict temp_buffer_start = (int16 __aie_dm_resource_c*) temp_buffer;
        auto tempIT = aie::begin_vector<BATCH_SIZE>(temp_buffer_start);
        int16 __aie_dm_resource_d* __restrict temp_buffer2_start = (int16 __aie_dm_resource_d*) temp_buffer2;
        auto temp2IT = aie::begin_vector<BATCH_SIZE>(temp_buffer2_start);
        int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*)out_data.data();
        auto OutIt = aie::begin_vector<BATCH_SIZE>(out_buffer_start);  

        tile_obj.set_rounding(aie::rounding_mode::floor);

        // 2cc per iteration (two SRS passes and two stores: the result and its sign)
        for (unsigned j=0; j < LENGTH / (2*BATCH_SIZE); j++) { 
            aie::vector<int16, BATCH_SIZE> terms = *InIt++;
            aie::vector<int16, BATCH_SIZE> mul_barret_vec = *OutIt++;
            aie::accum<acc48, BATCH_SIZE> mt_acc = aie::mul(terms, (int16)factor_neg);
                LOG_VECTOR(terms, true, "terms=");
                LOG_VECTOR(mt_acc.to_vector<int16>(0), true, "mt_acc as vec =");
                LOG_VECTOR(mul_barret_vec, true, "mul_barret_vec=");
            aie::accum<acc48, BATCH_SIZE> reduced_acc = aie::msc(mt_acc, (int16)MODULUS, mul_barret_vec); 
                LOG_VECTOR(reduced_acc, true, "reduced_acc=");
                LOG_VECTOR(reduced_acc.to_vector<int16>(0), true, "\nreduced_acc as vector=");
            *temp2IT++ = reduced_acc.to_vector<int16>(0);
            *tempIT++ = reduced_acc.to_vector<int16>(16);
        }
    }


    // Reduce to [0,q]
    // Only for first half of buffer
    {
        int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*)in_data.data();
        auto InIt = aie::begin_vector<BATCH_SIZE>(in_buffer_start);
        int16 __aie_dm_resource_c* __restrict temp_buffer_start = (int16 __aie_dm_resource_c*) temp_buffer;
        auto tempIT = aie::begin_vector<BATCH_SIZE>(temp_buffer_start);
        int16 __aie_dm_resource_d* __restrict temp_buffer2_start = (int16 __aie_dm_resource_d*) temp_buffer2;
        auto temp2IT = aie::begin_vector<BATCH_SIZE>(temp_buffer2_start);
        int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*)out_data.data();
        auto OutIt = aie::begin_vector<BATCH_SIZE>(out_buffer_start);  


        // 1cc per iteration
        for (unsigned j=0; j < LENGTH / (2*BATCH_SIZE); j++) { 
            aie::vector<int16, BATCH_SIZE> sign = *tempIT++;
                LOG_VECTOR(sign, true, "\nsign=");
            aie::vector<int16, BATCH_SIZE> terms = *temp2IT++;
            aie::accum<acc48, BATCH_SIZE> reduced_acc(terms);
                LOG_VECTOR(reduced_acc.to_vector<int16>(0), true, "\nreduced_acc as vector=");
            aie::accum<acc48, BATCH_SIZE> acc2 = aie::msc(reduced_acc, sign, (int16)MODULUS);
                LOG_VECTOR(acc2, true, "\nacc2=");
                LOG_VECTOR(acc2.to_vector<int16>(0), true, "\nacc2 as vector=");
            aie::vector<int16, BATCH_SIZE> reduced = acc2.to_vector<int16>(0);
                LOG_VECTOR(reduced, true, "\nreduced=");
            *OutIt++ = reduced;
        }
    }



    // Mul with twiddle factor + reduction to [-q,q] + determine sign
    // Only for second half of buffer
    {
        int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*)in_data.data() + (LENGTH>>1);
        auto InIt = aie::begin_vector<BATCH_SIZE>(in_buffer_start);
        int16 __aie_dm_resource_c* __restrict temp_buffer_start = (int16 __aie_dm_resource_c*) temp_buffer;
        auto tempIT = aie::begin_vector<BATCH_SIZE>(temp_buffer_start);
        int16 __aie_dm_resource_d* __restrict temp_buffer2_start = (int16 __aie_dm_resource_d*) temp_buffer2;
        auto temp2IT = aie::begin_vector<BATCH_SIZE>(temp_buffer2_start);
        int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*)out_data.data() + (LENGTH>>1);
        auto OutIt = aie::begin_vector<BATCH_SIZE>(out_buffer_start);  


        // 2cc per iteration (two SRS passes and two stores: the result and its sign)
        for (unsigned j=0; j < LENGTH / (2*BATCH_SIZE); j++) { 
            aie::vector<int16, BATCH_SIZE> terms = *InIt++;
            aie::vector<int16, BATCH_SIZE> mul_barret_vec = *OutIt++;
            aie::accum<acc48, BATCH_SIZE> mt_acc = aie::mul(terms, (int16)factor_neg);
                LOG_VECTOR(terms, true, "terms=");
                LOG_VECTOR(mt_acc.to_vector<int16>(0), true, "mt_acc as vec =");
                LOG_VECTOR(mul_barret_vec, true, "mul_barret_vec=");
            aie::accum<acc48, BATCH_SIZE> reduced_acc = aie::msc(mt_acc, (int16)MODULUS, mul_barret_vec); 
                LOG_VECTOR(reduced_acc, true, "reduced_acc=");
                LOG_VECTOR(reduced_acc.to_vector<int16>(0), true, "\nreduced_acc as vector=");
            *temp2IT++ = reduced_acc.to_vector<int16>(0);
            *tempIT++ = reduced_acc.to_vector<int16>(16);
        }
    }


    // Reduce to [0,q]
    // Only for second half of buffer
    {
        int16 __aie_dm_resource_a* __restrict in_buffer_start = (int16 __aie_dm_resource_a*)in_data.data();
        auto InIt = aie::begin_vector<BATCH_SIZE>(in_buffer_start);
        int16 __aie_dm_resource_c* __restrict temp_buffer_start = (int16 __aie_dm_resource_c*) temp_buffer;
        auto tempIT = aie::begin_vector<BATCH_SIZE>(temp_buffer_start);
        int16 __aie_dm_resource_d* __restrict temp_buffer2_start = (int16 __aie_dm_resource_d*) temp_buffer2;
        auto temp2IT = aie::begin_vector<BATCH_SIZE>(temp_buffer2_start);
        int16 __aie_dm_resource_b* __restrict out_buffer_start = (int16 __aie_dm_resource_b*)out_data.data() + (LENGTH>>1);
        auto OutIt = aie::begin_vector<BATCH_SIZE>(out_buffer_start);  

        // 1cc per iteration
        for (unsigned j=0; j < LENGTH / (2*BATCH_SIZE); j++) { 
            aie::vector<int16, BATCH_SIZE> sign = *tempIT++;
                LOG_VECTOR(sign, true, "\nsign=");
            aie::vector<int16, BATCH_SIZE> terms = *temp2IT++;
            aie::accum<acc48, BATCH_SIZE> reduced_acc(terms);
                LOG_VECTOR(reduced_acc.to_vector<int16>(0), true, "\nreduced_acc as vector=");
            aie::accum<acc48, BATCH_SIZE> acc2 = aie::msc(reduced_acc, sign, (int16)MODULUS);
                LOG_VECTOR(acc2, true, "\nacc2=");
                LOG_VECTOR(acc2.to_vector<int16>(0), true, "\nacc2 as vector=");
            aie::vector<int16, BATCH_SIZE> reduced = acc2.to_vector<int16>(0);
                LOG_VECTOR(reduced, true, "\nreduced=");
            *OutIt++ = reduced;
        }
    }

    
}


// Set round and saturation mode (although these should already be the defaults)
void set_modes_final_reduction(void) {
    aie::tile tile_obj=aie::tile::current(); //get the tile of the kernel
    tile_obj.set_rounding(aie::rounding_mode::symmetric_zero);
    tile_obj.set_saturation(aie::saturation_mode::none);
}
