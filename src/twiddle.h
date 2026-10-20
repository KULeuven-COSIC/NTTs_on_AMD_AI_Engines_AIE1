#pragma once

#include <adf.h>
#include "auxiliary.h"
#include "kernels/debug.h"
#include <algorithm>
#include <float.h>

// There are two kernels implemented
enum class NTT_KERNEL_TYPE {
    CT_NO_to_BO,
    GS_BO_to_NO
};



// Bit reverse
inline unsigned long long bitrev(const unsigned long long a, const unsigned short NO_OF_BITS) {
    unsigned long long rev = 0;
    for (unsigned short i = 0; i < NO_OF_BITS; i++)
        if ((a & (1 << i))) rev |= 1 << ((NO_OF_BITS - 1) - i);
    return rev;
}

// Round a float to a int16
// Custom function to not have to rely on std which might not be present
template<typename T>
inline T round(const long double a) {
    const T rounded = T(a + (long double)0.5);
    return rounded;
}

// Calculate the inverse of N modulo M
template<typename T>
inline T modular_inverse(T N, T M) {
     int a = N, b = M, x0 = 1, x1 = 0;

    while (b) {
        int q = a / b;
        int t = a % b;
        a = b;
        b = t;

        int temp = x1;
        x1 = x0 - q * x1;
        x0 = temp;
    }
    return (x0 % M + M) % M;
}


// Perform modular exponentiation
inline int modular_exponentation(const unsigned base, const unsigned power, const unsigned modulus) {
    unsigned res = 1;
    for (unsigned i = 0; i < power; i++) res = (res * base) % modulus;
    return res;    
}

// reduce to [-q/2,q/2]
template<typename T>
inline T red_neg(T a, T MODULUS) {
    a = a % MODULUS;
    if (a > (MODULUS - 1)/2) {
        a = a - MODULUS;
    } else if (a < -(MODULUS - 1)/2) {
        a = a + MODULUS;
    }
    return a;
}


// Compute the twiddle factors omega^i (or omega^-i for the inverse transform) for i = 0..ROOT_SIZE-1,
// reduced to [-(q-1)/2, (q-1)/2], in natural order (used by the radix-4 and the inverse radix-2 int16 pipelines)
template<uint16 MODULUS, uint16 ROOT, uint16 ROOT_SIZE, bool INVERSE_TRANSFORM>
inline void compute_twiddle_factors_kyber(int16* twiddle_factors) {

    LOG("\nROOT SIZE second%u: ", ROOT_SIZE);
    twiddle_factors[0] = 1;
    int16 cur = twiddle_factors[0];
    const int16 FACTOR = INVERSE_TRANSFORM ? modular_inverse(ROOT, MODULUS) : ROOT;
    for (unsigned long long i = 1; i < ROOT_SIZE; i++) {
      cur = int32(cur * FACTOR) % MODULUS;
      cur = red_neg(cur, (int16)MODULUS);
      LOG("cur: %d\n", cur);
      twiddle_factors[i] = cur;
    } 
}



// Compute the Barrett multipliers round(w * 2^16 / q) of the twiddle factors (K = 16)
template<uint16 MODULUS, uint16 ROOT_SIZE>
inline void compute_barrett_factors_kyber(int16* twiddle_factors, int16* barret_factors) {
    constexpr uint16 K_barret = 16;
    for (unsigned long long i = 0; i < ROOT_SIZE; i++) {
      const int16 rounded = round(float(twiddle_factors[i] * (1 << K_barret)) / float(MODULUS));
      LOG("twiddle precalc: %f, rounded : %d\n", float(twiddle_factors[i] * (1 << K_barret)) / float(MODULUS), rounded);
      barret_factors[i] = int16(rounded);
      LOG("twiddle : %d\n", twiddle_factors[i]);
      LOG("barret : %d\n", barret_factors[i]);
    } 
}




// Compute the twiddle factors (stored in bit-reversed order), used by the forward radix-2 pipelines
template<typename DATA_TYPE>
inline void compute_twiddle_factors(const DATA_TYPE MODULUS, const DATA_TYPE ROOT, const uint16 NO_OF_LAYERS, const int FACTORS_LENGTH, DATA_TYPE* twiddle_factors, bool inverse) {
    twiddle_factors[0] = 1;
    long long cur = twiddle_factors[0];
    const long long FACTOR = inverse ? (long long) modular_inverse(ROOT, MODULUS) : (long long) ROOT;
    for (unsigned long long i = 1; i < FACTORS_LENGTH; i++) {
      long long mul = cur * FACTOR;
      cur = red_neg(mul, (long long)MODULUS);
      LOG("final cur : %lld\n", cur);
      twiddle_factors[bitrev(i, NO_OF_LAYERS-1)] = DATA_TYPE(cur); // NO_LAYERS = log_2(FACTORS_LENGTH)
    } 
}



// Compute the Barrett multipliers round(w * 2^K / q) of the twiddle factors
template<typename DATA_TYPE>
inline void compute_barret_factors(const DATA_TYPE MODULUS, const int FACTORS_LENGTH, const int K_BARRETT, DATA_TYPE* twiddle_factors, DATA_TYPE* barret_factors) {
    for (unsigned long long i = 0; i < FACTORS_LENGTH; i++) {
      const DATA_TYPE rounded = round<DATA_TYPE>((long double)((long long)(twiddle_factors[i]) * (1LL << K_BARRETT)) / (long double)(MODULUS));
      LOG("twiddle: %d, double : %Lf, double shifted: %Lf\n", twiddle_factors[i], (long double)(twiddle_factors[i]), (long double)((long long)(twiddle_factors[i]) * (1LL << K_BARRETT)));
      LOG("twiddle precalc: %Lf, rounded : %d\n", (long double)((long long)(twiddle_factors[i]) * (1LL << K_BARRETT)) / (long double)(MODULUS), rounded);
      LOG("twiddle precalc: %Lf, rounded : %d\n", (long double)((long long)(twiddle_factors[i]) * (1LL << K_BARRETT)) / (long double)(MODULUS));
      LOG("Mantissa bits in long double: %d\n", LDBL_MANT_DIG);
      barret_factors[i] = DATA_TYPE(rounded);
      LOG("twiddle : %d\n", twiddle_factors[i]);
      LOG("barret : %d\n", barret_factors[i]);
    } 
}



// Expand the factors so they can be read in contiguously
template<typename DATA_TYPE>
inline void expand_factors(const unsigned LENGTH, DATA_TYPE* twiddle_factors, DATA_TYPE* barret_factors, DATA_TYPE* factors_combined, unsigned int layer) {

    // invar : repetitions * no_factors = NTT_LENGTH/2
    const unsigned long long repetitions = LENGTH>>(layer+1);
    const unsigned long long no_factors = 1<<layer;


    // Expand the factors
    DATA_TYPE* twiddle_factors_expanded = factors_combined;
    DATA_TYPE* barret_factors_expanded = factors_combined + (LENGTH>>1);  
    

    // Factors to duplicate = first [no_factors] of the input array
    // Duplicated [repetitions] times
    // This can be done faster: first copy from original to this buffer, then copy from start of this buffer (can copy two at once, then four, then eight, ...)
    for (unsigned long long j = 0; j < repetitions; j++) {
        std::memcpy((void*)(twiddle_factors_expanded + j*no_factors), (void*)twiddle_factors, no_factors*sizeof(DATA_TYPE));
        std::memcpy((void*)(barret_factors_expanded + j*no_factors), (void*)barret_factors, no_factors*sizeof(DATA_TYPE));
    }


    // Print to check
    LOG("\nexpanded factors for layer %d: ", layer);
    for (unsigned long long i = 0; i < LENGTH>>1; i++) {
        LOG(" (%d,%d)", twiddle_factors_expanded[i], barret_factors_expanded[i]);
    }
    LOG("\n\n");
    
}



// Expand the factors so they can be read in contiguously
// For the CT_BO_TO_NO transform
template<typename DATA_TYPE>
inline void expand_twiddle_and_barrett_factors(const unsigned LENGTH, DATA_TYPE* twiddle_factors, DATA_TYPE* barret_factors, DATA_TYPE* twiddle_factors_expanded, DATA_TYPE* barrett_multipliers_expanded, unsigned int layer) {

    // invar : repetitions * no_factors = NTT_LENGTH/2
    const unsigned long long repetitions = LENGTH>>(layer+1);
    const unsigned long long no_factors = 1<<layer;
    

    // Expand the factors: the twiddle factors must be repeated in pairs!
    // e.g. for layer=2 : 0,2,0,2,4,6,4,6

    // Pointer to where we write the expanded values
    unsigned write_index = 0;

    for (unsigned g = 0; g < no_factors/2; ++g) {
        // Indices in the input arrays
        unsigned idx1 = g * 2 * repetitions;
        unsigned idx2 = idx1 + repetitions;

        // Fetch the two base values for this group
        DATA_TYPE tw1 = twiddle_factors[idx1];
        DATA_TYPE tw2 = twiddle_factors[idx2];
        DATA_TYPE br1 = barret_factors[idx1];
        DATA_TYPE br2 = barret_factors[idx2];


        // Repeat the pair `repetitions` times
        for (unsigned r = 0; r < repetitions; ++r) {
            twiddle_factors_expanded[write_index]     = tw1;
            barrett_multipliers_expanded[write_index] = br1;
            ++write_index;

            twiddle_factors_expanded[write_index]     = tw2;
            barrett_multipliers_expanded[write_index] = br2;
            ++write_index;
        }
    }


    // Print to check
    LOG("\nexpanded factors for layer %d: ", layer);
    for (unsigned long long i = 0; i < LENGTH>>1; i++) {
        LOG(" (%d,%d)", twiddle_factors_expanded[i], barrett_multipliers_expanded[i]);
    }
    LOG("\n\n");
    
}


// Fill in the factor vectors for every layer
template<typename DATA_TYPE, unsigned LENGTH, DATA_TYPE MODULUS, DATA_TYPE ROOT, unsigned K_BARRETT, NTT_KERNEL_TYPE KERNEL_TYPE, bool INVERSE_TRANSFORM, uint16 NO_OF_LAYERS, uint16 MAX_NO_OF_LAYERS>
inline void determine_factors(std::vector<std::vector<DATA_TYPE>>& factors) {
  // Determine twiddle factors
  const int FACTORS_LENGTH = (1 << (MAX_NO_OF_LAYERS - 1)); // 2^{NO_OF_LAYERS-1}
  DATA_TYPE *twiddle_factors= new DATA_TYPE[FACTORS_LENGTH];
  DATA_TYPE *barret_factors= new DATA_TYPE[FACTORS_LENGTH];

  compute_twiddle_factors(MODULUS, ROOT, MAX_NO_OF_LAYERS, FACTORS_LENGTH, twiddle_factors, INVERSE_TRANSFORM); 
  compute_barret_factors(MODULUS, FACTORS_LENGTH, K_BARRETT, twiddle_factors, barret_factors);

  for (unsigned int i = 0; i < NO_OF_LAYERS; i++) {
    // Generate the expanded and combined factors
    const int layer = KERNEL_TYPE == NTT_KERNEL_TYPE::CT_NO_to_BO ? i : MAX_NO_OF_LAYERS - i - 1; // For the GS_BO_to_NO kernel, the layers are in reversed order w.r.t. the factors
    LOG("layer %d", layer);
    factors[i].resize(LENGTH);
    expand_factors(LENGTH, twiddle_factors, barret_factors, &factors[i][0], layer);
  }
}



// Fill in the factor vectors for every layer
template<typename DATA_TYPE, unsigned LENGTH, DATA_TYPE MODULUS, DATA_TYPE ROOT, unsigned K_BARRETT, NTT_KERNEL_TYPE KERNEL_TYPE, bool INVERSE_TRANSFORM, uint16 NO_OF_LAYERS, uint16 MAX_NO_OF_LAYERS>
inline void determine_twiddle_and_barrett(std::vector<std::vector<DATA_TYPE>>& twiddle_factors_layers, std::vector<std::vector<DATA_TYPE>>& barrett_multipliers_layers) {
  // Determine twiddle factors
  const int FACTORS_LENGTH = (1 << (MAX_NO_OF_LAYERS - 1)); // 2^{NO_OF_LAYERS-1}
  DATA_TYPE *twiddle_factors= new DATA_TYPE[LENGTH];
  DATA_TYPE *barret_factors= new DATA_TYPE[LENGTH];

  compute_twiddle_factors_kyber<MODULUS, ROOT, LENGTH, INVERSE_TRANSFORM>(twiddle_factors); 
  compute_barrett_factors_kyber<MODULUS, LENGTH>(twiddle_factors, barret_factors);

  for (unsigned int i = 0; i < NO_OF_LAYERS; i++) {
    // Generate the expanded and combined factors
    const int layer = KERNEL_TYPE == NTT_KERNEL_TYPE::CT_NO_to_BO ? i : MAX_NO_OF_LAYERS - i - 1; // For the GS_BO_to_NO kernel, the layers are in reversed order w.r.t. the factors
    LOG("layer %d, %d, %d", layer, twiddle_factors_layers.size(), barrett_multipliers_layers.size());
    twiddle_factors_layers[i].resize(LENGTH/2);
    barrett_multipliers_layers[i].resize(LENGTH/2);
    expand_twiddle_and_barrett_factors(LENGTH, twiddle_factors, barret_factors, &twiddle_factors_layers[i][0], &barrett_multipliers_layers[i][0], layer);
  }
}




// Radix-8 (Section 5.3): factors of the 8x8 butterfly matrix (twiddle_factors_layers) and the
// between-layer twiddle factors with their 32-bit Barrett multipliers (K = 31) for the reduction tile
template<uint16 MODULUS, uint16 ROOT, uint16 LENGTH>
inline void determine_factors_r8(std::vector<std::vector<int16>>& twiddle_factors_layers, std::vector<std::vector<int16>>& twiddle_factors_reduction, std::vector<std::vector<int32>>& barrett_factors_reduction) {

  constexpr unsigned SUBNTT_SIZE = 8;

{
  // Determine twiddle factors
  int16 *twiddle_factors= new int16[SUBNTT_SIZE];
  constexpr uint16 ROOT_8 = Power<ROOT, LENGTH / SUBNTT_SIZE, MODULUS>::value;
      LOG("ROOT_8 %d", ROOT_8);
    twiddle_factors[0] = 1;
    int16 cur = twiddle_factors[0];
    for (unsigned long long i = 1; i < SUBNTT_SIZE; i++) {
      cur = uint32(cur * ROOT_8) % MODULUS;
      twiddle_factors[i] = cur;
    } 

  // Determine barrett factors
  int16 *barrett_factors= new int16[SUBNTT_SIZE];
  constexpr uint16 K_barret = 15;
  for (unsigned long long i = 0; i < SUBNTT_SIZE; i++) {
        const int16 rounded = round(float(twiddle_factors[i] * (1 << K_barret)) / float(MODULUS));
        LOG("twiddle precalc: %f, rounded : %d\n", float(twiddle_factors[i] * (1 << K_barret)) / float(MODULUS), rounded);
        barrett_factors[i] = int16(rounded);
        LOG("twiddle : %d\n", twiddle_factors[i]);
        LOG("barret : %d\n", barrett_factors[i]);
    } 


  // Twiddle vector must look like this:
  /*

    ij = input i, output j -> omega^(ij)

    first four MACs (i.e. first four inputs), first iteration (i.e. first four outputs): 
        00, 10, 20, 30 | 01, 11, 21, 31 | 02, 12, 22, 32 | 03, 13, 23, 33
    first four MACs (i.e. first four inputs), second iteration (i.e. second four outputs): 
        04, 14, 24, 34 | 05, 15, 25, 35 | 06, 16, 26, 36 | 07, 17, 27, 37 
    second four MACs (i.e. second four inputs), first iteration (i.e. first four outputs): 
        40, 50, 60, 70 | 41, 51, 61, 71 | 42, 52, 62, 72 | 43, 53, 63, 73 
    second four MACs (i.e. second four inputs), second iteration (i.e. second four outputs): 
        44, 54, 64, 74 | 45, 55, 65, 75 | 46, 56, 66, 76 | 47, 57, 67, 77

    Additionally, each group of four factors is bit-reversed. This is because the inputs are ordered bit-reversed in the mul8/mac8 intrinsic due to the permutation constraints

  */


  // Generate the factors for the 8-NTTs
  constexpr unsigned MAC_PER_CYCLE = 4; // number of consecutive twiddle factors = number of MAC per output per iteration
  constexpr unsigned FIRST_OR_SECOND_FOUR_INPUTS = SUBNTT_SIZE / MAC_PER_CYCLE; // first four or second four MACs / inputs
  constexpr unsigned FIRST_OR_SECOND_FOUR_OUTPUTS = SUBNTT_SIZE / MAC_PER_CYCLE; // first four or second four outputs
  twiddle_factors_layers[0].resize(SUBNTT_SIZE*SUBNTT_SIZE);
  for (unsigned i = 0; i < FIRST_OR_SECOND_FOUR_INPUTS; i++) {
    for (unsigned j = 0; j < FIRST_OR_SECOND_FOUR_OUTPUTS; j++) {
        // inside one vector:
        for (unsigned k = 0; k < 4; k++) {
            for (unsigned l = 0; l < MAC_PER_CYCLE; l++) {
                const unsigned input = 4 * i + l;
                const unsigned output = 4 * j + k;
                const int16 factor = twiddle_factors[(output * input) % SUBNTT_SIZE];
                const int16 signed_factor = factor < (MODULUS-1)/2 ? factor : factor-MODULUS;
                LOG("input: %d, ", input);
                LOG("output: %d, ", output);

                LOG("factor: %d, %d\n", factor, signed_factor);
                const unsigned consecutive_position = bitrev(l, 2); // the inputs are ordered differently in the mac due to the permutation constraints
                twiddle_factors_layers[0][32*i + 16*j + 4*k + consecutive_position] = signed_factor;
            }
        }
    }
  }

    LOG("factors for layer:");
  for (unsigned j = 0; j < SUBNTT_SIZE * SUBNTT_SIZE; j++) {
      LOG("%d, ", twiddle_factors_layers[0][j]);
  }
}





{

  // Determine twiddle factors
  int16 *twiddle_factors= new int16[LENGTH];
    twiddle_factors[0] = 1;
    int16 cur = twiddle_factors[0];
    for (unsigned long long i = 1; i < LENGTH; i++) {
        cur = uint32(cur * ROOT) % MODULUS;
        LOG("cur: %d\n", cur);
        twiddle_factors[i] = cur;
    } 

  // Determine barrett factors
  int32 *barrett_factors= new int32[LENGTH];
  constexpr uint16 K_mul = 31;
  for (unsigned long long i = 0; i < LENGTH; i++) {
        const int32 rounded = round(float(int64(twiddle_factors[i]) << K_mul) / float(MODULUS));
        LOG("twiddle precalc: %f, rounded : %d\n", float(int64(twiddle_factors[i]) << K_mul) / float(MODULUS), rounded);
        barrett_factors[i] = int32(rounded);
        LOG("twiddle : %d\n", twiddle_factors[i]);
        LOG("barret : %d\n", barrett_factors[i]);
    } 

  twiddle_factors_reduction[0].resize(LENGTH);
  barrett_factors_reduction[0].resize(LENGTH);
    for (unsigned i = 0; i < LENGTH / SUBNTT_SIZE; i++) {
        for (unsigned j = 0; j < SUBNTT_SIZE; j++) {
            const unsigned group = unsigned(i / 2); // At the reduction phase, the output sets are interleaved per two 8-NTTs
            const unsigned index_in_group = i % 2;
            
            twiddle_factors_reduction[0][2 * group * SUBNTT_SIZE + j * 2 + index_in_group] = twiddle_factors[(i * j) % LENGTH];
            barrett_factors_reduction[0][2 * group * SUBNTT_SIZE + j * 2 + index_in_group] = barrett_factors[(i * j) % LENGTH];
        }
    }


    LOG("factors for reduction:");
  for (unsigned j = 0; j < LENGTH; j++) {
      LOG("(%d, %d) ", twiddle_factors_reduction[0][j], barrett_factors_reduction[0][j]);
  }
}
    
  
}

// Radix-4 (Section 5.2): twiddle factors and Barrett multipliers of one radix-4 layer.
// factors_combined = [twiddle factors (LENGTH) | Barrett multipliers (LENGTH)], one entry per input coefficient
template<typename DATA_TYPE, unsigned LENGTH, unsigned MODULUS, unsigned ROOT_SIZE, unsigned MAX_NO_OF_R4_LAYERS>
inline void expand_factors_layer_r4(DATA_TYPE* twiddle_factors, DATA_TYPE* barrett_factors, DATA_TYPE* factors_combined, const int layer) {

    constexpr unsigned LOG2_RADIX = 2; // number of radix-2 layers per radix°4 layer
    constexpr unsigned RADIX = 1<<LOG2_RADIX;
    const int N = std::pow(RADIX, layer+1);
    const int OMEGA_POWER = modular_exponentation(RADIX, MAX_NO_OF_R4_LAYERS-layer-1, MODULUS); // such that omega_{ROOT_SIZE}^{OMEGA_POWER} = omega_N
    const int NO_OF_BITS = layer * LOG2_RADIX; // for bitrev of twiddle factors, is equal to log2(N / RADIX)
    LOG("\nlayer %d, RADIX: %d, N: %d, MAX_NO_OF_R4_LAYERS: %d, MODULUS: %d, OMEGA_POWER: %d, NO_OF_BITS; %d", layer, RADIX, N, MAX_NO_OF_R4_LAYERS, MODULUS, OMEGA_POWER, NO_OF_BITS);

    // insert the factors
    for (unsigned n1 = 0; n1 < RADIX; n1++) {
        for (unsigned ntt_index = 0; ntt_index < LENGTH / RADIX; ntt_index++) {
            const int k2 = ntt_index % int(N / RADIX);
            const int index = int(LENGTH / RADIX) * n1 + ntt_index;
            const int power = (OMEGA_POWER * n1 * bitrev(k2, NO_OF_BITS)) % ROOT_SIZE;
            factors_combined[index] = twiddle_factors[power];
            factors_combined[LENGTH + index] = barrett_factors[power]; 
            LOG("\nn1: %d, ntt_index: %d, k2: %d, index: %d, power: %d, twiddle: %d, barrett: %d", n1, ntt_index, k2, index, power, twiddle_factors[power], barrett_factors[power]);
        }
    }

    LOG("\nfactors for layer %d:", layer);
    for (unsigned j = 0; j < LENGTH; j++) {
      LOG("(%d, %d) ", factors_combined[j], factors_combined[j+LENGTH]);
    }
}


template<typename DATA_TYPE, unsigned LENGTH, DATA_TYPE MODULUS, DATA_TYPE ROOT, unsigned ROOT_SIZE, unsigned K_BARRETT, NTT_KERNEL_TYPE KERNEL_TYPE, bool INVERSE_TRANSFORM, uint16 NO_OF_R4_LAYERS>
inline void determine_factors_radix4(std::vector<std::vector<DATA_TYPE>>& factors) {


  // Determine twiddle and barrett factors
  int16 *twiddle_factors= new int16[ROOT_SIZE];
  int16 *barrett_factors= new int16[ROOT_SIZE];
  compute_twiddle_factors_kyber<MODULUS, ROOT, ROOT_SIZE, INVERSE_TRANSFORM>(twiddle_factors); 
  compute_barrett_factors_kyber<MODULUS, ROOT_SIZE>(twiddle_factors, barrett_factors);


  // Generate the expanded and combined factors
  constexpr bool NEGATIVE_WRAPPED = false;
  constexpr unsigned RADIX = 4;
  constexpr unsigned MAX_NO_OF_R4_LAYERS = LOG2<ROOT_SIZE>::value / 2;
  for (unsigned int i = 0; i < NO_OF_R4_LAYERS; i++) {
    const int layer = KERNEL_TYPE == NTT_KERNEL_TYPE::CT_NO_to_BO ? i : MAX_NO_OF_R4_LAYERS - i - 1; // For the GS_BO_to_NO kernel, the layers are in reversed order w.r.t. the factors


    factors[i].resize(LENGTH*2);
    expand_factors_layer_r4<DATA_TYPE, LENGTH, MODULUS, ROOT_SIZE, MAX_NO_OF_R4_LAYERS>(twiddle_factors, barrett_factors, &factors[i][0], layer);
  }
}
