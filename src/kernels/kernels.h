
#pragma once

/*
 * Declarations of all AI Engine kernels used in this artifact.
 *
 * Kernels that need precomputed constants (twiddle factors, Barrett multipliers)
 * or scratch buffers are implemented as kernel classes: the constants and
 * scratch buffers are registered as kernel parameters (REGISTER_PARAMETER) so the
 * AIE compiler places them in the local data memory of the tile. Kernels
 * without state are plain (templated) functions.
 *
 * Naming conventions used throughout:
 *   NO = natural order, BO = bit-reversed order
 *   CT = Cooley-Tukey (decimation in time), GS = Gentleman-Sande (decimation in frequency)
 */



/* ------------------------------------------------------------------------- */
/* Radix-2, int16 (Section 5.1)                                              */
/* ------------------------------------------------------------------------- */

// One forward constant-geometry radix-2 layer (CT, NO input, BO output).
// factors = [twiddle factors (LENGTH/2) | Barrett multipliers (LENGTH/2)]
// Implementation: kernels/ntt_r2_kernels.cc
template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
class NTT_CT_NO_TO_BO_IMPROVED_KERNEL_CLASS
{
private:
  alignas(16) int16 (&factors)[LENGTH];
  alignas(16) int16 (&temp_buffer)[LENGTH>>1];
public:
    NTT_CT_NO_TO_BO_IMPROVED_KERNEL_CLASS( int16 (&factors_)[LENGTH], int16 (&temp_buffer_)[LENGTH>>1]);
    void NTT_CT_NO_to_BO(adf::input_buffer<int16> & in_data, adf::output_buffer<int16> & out_data);
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(NTT_CT_NO_TO_BO_IMPROVED_KERNEL_CLASS::NTT_CT_NO_to_BO);
        REGISTER_PARAMETER( factors );
        REGISTER_PARAMETER( temp_buffer );
    }
};


// One inverse constant-geometry radix-2 layer (BO input, NO output, unshuffle after every layer).
// LAYER is the index of the layer in the pipeline (the first and last layer are specialised).
// Implementation: kernels/ntt_r2_kernels.cc
template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE, unsigned NO_OF_LAYERS, unsigned LAYER>
class NTT_CT_BO_TO_NO_IMPROVED_KERNEL_CLASS
{
private:
  alignas(16) int16 (&twiddle_factors)[LENGTH/2];
  alignas(16) int16 (&barrett_multipliers)[LENGTH/2];
  alignas(16) int16 (&temp_buffer)[LENGTH/2];
  alignas(16) int16 (&temp_buffer2)[LENGTH/4];
public:
    NTT_CT_BO_TO_NO_IMPROVED_KERNEL_CLASS(int16 (&twiddle_factors_)[LENGTH/2], int16 (&barrett_multipliers_)[LENGTH/2], int16 (&temp_buffer_)[LENGTH/2], int16 (&temp_buffer2_)[LENGTH/4]);
    void NTT_CT_BO_to_NO(adf::input_buffer<int16> & in_data, adf::output_buffer<int16> & out_data);
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(NTT_CT_BO_TO_NO_IMPROVED_KERNEL_CLASS::NTT_CT_BO_to_NO);
        REGISTER_PARAMETER( twiddle_factors );
        REGISTER_PARAMETER( barrett_multipliers );
        REGISTER_PARAMETER( temp_buffer );
        REGISTER_PARAMETER( temp_buffer2 );
    }
};



/* ------------------------------------------------------------------------- */
/* Radix-2, int32 (Section 5.1, int32 variant)                               */
/* ------------------------------------------------------------------------- */

// One forward radix-2 layer on 32-bit coefficients (CT, NO input, BO output).
// factors = [twiddle factors (LENGTH/2) | Barrett multipliers (LENGTH/2)]
// Implementation: kernels/ntt_r2_int32_kernels.cc
template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
class NTT_INT32_CT_NO_TO_BO_KERNEL_CLASS
{
private:
  alignas(16) int32 (&factors)[LENGTH];
  alignas(16) int32 (&temp_buffer)[LENGTH>>1];
public:
    NTT_INT32_CT_NO_TO_BO_KERNEL_CLASS( int32 (&factors_)[LENGTH], int32 (&temp_buffer_)[LENGTH>>1]);
    void NTT_CT_NO_to_BO(adf::input_buffer<int32> & in_data, adf::output_buffer<int32> & out_data);
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(NTT_INT32_CT_NO_TO_BO_KERNEL_CLASS::NTT_CT_NO_to_BO);
        REGISTER_PARAMETER( factors );
        REGISTER_PARAMETER( temp_buffer );
    }
};


// One inverse radix-2 layer on 32-bit coefficients (GS, BO input, NO output).
// factors = [twiddle factors (LENGTH/2) | Barrett multipliers (LENGTH/2)]
// Implementation: kernels/ntt_r2_int32_kernels.cc
template<unsigned LENGTH, unsigned MODULUS, unsigned VECTOR_SIZE>
class NTT_INT32_GS_BO_TO_NO_KERNEL_CLASS
{
private:
  alignas(16) int32 (&factors)[LENGTH];
  alignas(16) int32 (&temp_buffer)[LENGTH>>1];
public:
    NTT_INT32_GS_BO_TO_NO_KERNEL_CLASS( int32 (&factors_)[LENGTH], int32 (&temp_buffer_)[LENGTH>>1]);
    void NTT_GS_BO_to_NO(adf::input_buffer<int32> & in_data, adf::output_buffer<int32> & out_data);
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(NTT_INT32_GS_BO_TO_NO_KERNEL_CLASS::NTT_GS_BO_to_NO);
        REGISTER_PARAMETER( factors );
        REGISTER_PARAMETER( temp_buffer );
    }
};



/* ------------------------------------------------------------------------- */
/* Radix-4, int16 (Section 5.2)                                              */
/* ------------------------------------------------------------------------- */

// One forward constant-geometry radix-4 layer (CT, NO input, BO output).
// factors = [twiddle factors (LENGTH) | Barrett multipliers (LENGTH)]
// ROOT_4 is a primitive 4-th root of unity modulo MODULUS (omega_4 in Equation 6/7).
// Implementation: kernels/ntt_r4_kernels.cc
template<unsigned LENGTH, unsigned MODULUS, unsigned ROOT_4, unsigned VECTOR_SIZE>
class NTT_R4_CT_NO_TO_BO_KERNEL_CLASS
{
private:
  alignas(16) int16 (&factors)[2*LENGTH];
  alignas(16) int16 (&temp_buffer)[LENGTH>>1];
public:
    NTT_R4_CT_NO_TO_BO_KERNEL_CLASS( int16 (&factors_)[2*LENGTH], int16 (&temp_buffer_)[LENGTH>>1]);
    void NTT_CT_NO_to_BO(adf::input_buffer<int16> & in_data, adf::output_buffer<int16> & out_data);
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(NTT_R4_CT_NO_TO_BO_KERNEL_CLASS::NTT_CT_NO_to_BO);
        REGISTER_PARAMETER( factors );
        REGISTER_PARAMETER( temp_buffer );
    }
};



/* ------------------------------------------------------------------------- */
/* Radix-8, int16 (Section 5.3)                                              */
/* ------------------------------------------------------------------------- */

// Block 1: radix-8 butterflies without reduction (int16 in, int32 out).
// Implementation: kernels/ntt_r8_kernels.cc
template<unsigned MODULUS, unsigned LENGTH, unsigned VECTOR_SIZE>
class NTT_R8_LAYERS_KERNEL_CLASS
{
public:
    static constexpr unsigned TWIDDLE_FACTORS_LENGTH = 64;
private:
  alignas(16) int16 (&twiddle_factors)[TWIDDLE_FACTORS_LENGTH];
  alignas(16) int32 (&temp_buffer2)[LENGTH];
public:
    NTT_R8_LAYERS_KERNEL_CLASS(int16 (&twiddle_factors_)[TWIDDLE_FACTORS_LENGTH], int32 (&temp_buffer2_)[LENGTH]);
    void NTT_R8_LAYERS(adf::input_buffer<int16> & in_data, adf::output_buffer<int32> & out_data) __attribute__((noinline));
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(NTT_R8_LAYERS_KERNEL_CLASS::NTT_R8_LAYERS);
        REGISTER_PARAMETER( twiddle_factors );
        REGISTER_PARAMETER( temp_buffer2 );
    }
};


// Block 2: 32-to-16-bit Barrett multiplication with the between-layer twiddle factors (int32 in, int16 out).
// Implementation: kernels/ntt_r8_kernels.cc
template<unsigned MODULUS, unsigned LENGTH, unsigned VECTOR_SIZE>
class NTT_R8_RECUTION_KERNEL_CLASS
{
private:
  alignas(16) int16 (&temp_buffer)[LENGTH];
  alignas(16) int32 (&temp_buffer2)[LENGTH>>1];
  alignas(16) int32 (&barrett_factors)[LENGTH];
  alignas(16) int16 (&twiddle_factors)[LENGTH];
public:
    NTT_R8_RECUTION_KERNEL_CLASS(int32 (&barrett_factors_)[LENGTH], int16 (&twiddle_factors_)[LENGTH], int16 (&temp_buffer_)[LENGTH], int32 (&temp_buffer2_)[LENGTH>>1]);
    void NTT_R8_REDUCTION(adf::input_buffer<int32> & in_data, adf::output_buffer<int16> & out_data) __attribute__((noinline));
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(NTT_R8_RECUTION_KERNEL_CLASS::NTT_R8_REDUCTION);
        REGISTER_PARAMETER( barrett_factors );
        REGISTER_PARAMETER( twiddle_factors );
        REGISTER_PARAMETER( temp_buffer );
        REGISTER_PARAMETER( temp_buffer2 );
    }
};



/* ------------------------------------------------------------------------- */
/* Full Barrett multiplication by a constant, int16 (Section 4.2, Table 1)   */
/* ------------------------------------------------------------------------- */

// Computes MULTIPLICATION_FACTOR * a mod MODULUS for every coefficient, fully reduced to [0, q).
// Implementation: kernels/full_reduction_kernels.cc
template<unsigned LENGTH, unsigned MODULUS, unsigned MULTIPLICATION_FACTOR>
class FULL_BARRETT_MULTIPLICATION_KERNEL_CLASS
{
private:
  alignas(16) int16 (&temp_buffer)[LENGTH>>1];
  alignas(16) int16 (&temp_buffer2)[LENGTH>>1];
public:
    FULL_BARRETT_MULTIPLICATION_KERNEL_CLASS( int16 (&temp_buffer_)[LENGTH>>1], int16 (&temp_buffer2_)[LENGTH>>1]);
    void full_barrett_multiplication_kernel(adf::input_buffer<int16> & __restrict in_data,
        adf::output_buffer<int16> & __restrict out_data) __attribute__((noinline));
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(FULL_BARRETT_MULTIPLICATION_KERNEL_CLASS::full_barrett_multiplication_kernel);
        REGISTER_PARAMETER( temp_buffer );
        REGISTER_PARAMETER( temp_buffer2 );
    }
};



/* ------------------------------------------------------------------------- */
/* Initialisation functions (set rounding and saturation mode of the tile)   */
/* ------------------------------------------------------------------------- */

// Defined in the source file of every NTT layer kernel (each tile is compiled separately)
void set_modes_ntt_layer(void);
// Defined in kernels/full_reduction_kernels.cc
void set_modes_final_reduction(void);



/* ------------------------------------------------------------------------- */
/* Bit-reversal unit, int16 (Section 6)                                      */
/* ------------------------------------------------------------------------- */

// Implementation: kernels/bitreverse_kernels.cc. LENGTH must be 256, 512, 1024, 2048 or 4096.
template<unsigned LENGTH>
void bitreverse_improved_kernel(adf::input_buffer<int16> & in_data,
        adf::output_buffer<int16> & out_data);



/* ------------------------------------------------------------------------- */
/* Input/output kernels                                                      */
/* ------------------------------------------------------------------------- */

// The test harness streams the polynomial in through four 128-bit PLIOs (one per quarter of the
// coefficients) and out through four 128-bit PLIOs. These kernels gather the four quarters into
// one LENGTH-sized buffer (and scatter it again at the output), so that the NTT pipeline itself
// only communicates through shared memory buffers.
// Implementation: kernels/io_kernels.cc
template<unsigned LENGTH>
void INPUT_KERNEL_INT16(adf::input_buffer<int16> & __restrict in_data_q0, adf::input_buffer<int16> & __restrict in_data_q1, adf::input_buffer<int16> & __restrict in_data_q2, adf::input_buffer<int16> & __restrict in_data_q3, adf::output_buffer<int16>& __restrict out_data);
template<unsigned LENGTH>
void OUTPUT_KERNEL_INT16(adf::input_buffer<int16>& __restrict in_data, adf::output_buffer<int16> & __restrict out_data_q0, adf::output_buffer<int16> & __restrict out_data_q1, adf::output_buffer<int16> & __restrict out_data_q2, adf::output_buffer<int16> & __restrict out_data_q3);

template<unsigned LENGTH>
void INPUT_KERNEL_INT32(adf::input_buffer<int32> & __restrict in_data_q0, adf::input_buffer<int32> & __restrict in_data_q1, adf::input_buffer<int32> & __restrict in_data_q2, adf::input_buffer<int32> & __restrict in_data_q3, adf::output_buffer<int32>& __restrict out_data);
template<unsigned LENGTH>
void OUTPUT_KERNEL_INT32(adf::input_buffer<int32>& __restrict in_data, adf::output_buffer<int32> & __restrict out_data_q0, adf::output_buffer<int32> & __restrict out_data_q1, adf::output_buffer<int32> & __restrict out_data_q2, adf::output_buffer<int32> & __restrict out_data_q3);
