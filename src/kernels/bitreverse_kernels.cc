
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
 * Vector-based bit-reversal unit for int16 (Section 6 of the paper), for LENGTH = 2^N, N = 8..12.
 *
 * The bit-reversal is decomposed into perfect-shuffle levels performed with in-place shuffle
 * butterflies (Figure 5b). Two shuffle levels are performed per loop: every iteration loads a group
 * of four vectors, interleaves them with two VSEL (select32) instructions and stores four vectors.
 * Two such loops perform the last four shuffle levels (M = 4 for 16-lane vectors). A last loop
 * performs the remaining "local" vector moves (vmd) that reverse the bits N-M-1 down to M; for
 * N = 8 and N = 9 this is the identity and the loop is a plain copy.
 */



// vmd = Vector Move with Delta: copy one vector to the output, displaced by delta vectors
#define vmd(delta) do { \
    v16int16 chess_storage(wr0) v1 = *inDataIt++; \
    *((outDataIt++)+(delta)) = v1; \
} while(0);


// Two in-place perfect-shuffle levels on four vectors that are DELTA vectors apart
#define two_layer_ilv() do {  \
    v16int16 chess_storage(wr0) v1 = *inDataIt++; \
    v16int16 chess_storage(wr1) v2 = *(inDataIt+DELTA-1); \
    v16int16 chess_storage(wr2) v3 = *(inDataIt+2*DELTA-1);  \
    v16int16 chess_storage(wr3) v4 = *(inDataIt+3*DELTA-1); \
 \
        v32int16 chess_storage(xc) ilv1 = select32(0b00111100001111000011110000111100, concat(v1,v2, v3, v4), 0, 0xf9f1f8f0, 0xfbf3faf2, 0x3120, 0, 0xf1f9f0f8, 0xf3fbf2fa, 0x2031); \
        v32int16 chess_storage(xd) ilv2 = select32(0b00111100001111000011110000111100, concat(v1,v2, v3, v4), 0, 0xfdf5fcf4, 0xfff7fef6, 0x3120, 0, 0xf5fdf4fc, 0xf7fff6fe, 0x2031); \
 \
    *outDataIt++ = ext_w(ilv1,0); \
    *(outDataIt+DELTA-1) = ext_w(ilv2,0); \
    *(outDataIt+2*DELTA-1) = ext_w(ilv1,1); \
    *(outDataIt+3*DELTA-1) = ext_w(ilv2,1); \
} while(0);




template<unsigned LENGTH>
void bitreverse_improved_kernel(adf::input_buffer<int16> & __restrict in_data, 
        adf::output_buffer<int16> & __restrict out_data) {

     static_assert(LENGTH == 256 || LENGTH == 512 || LENGTH == 1024 || LENGTH == 2048 || LENGTH == 4096, "LENGTH must be 256,512,1024,2048 or 4096");


    int16 __aie_dm_resource_a* __restrict in_data_start = (int16 __aie_dm_resource_a*)in_data.data();
    int16 __aie_dm_resource_a* __restrict in_data_start2 = (int16 __aie_dm_resource_a*)in_data.data();
    int16 __aie_dm_resource_b* __restrict out_data_start = (int16 __aie_dm_resource_b*)out_data.data();
    int16 __aie_dm_resource_b* __restrict out_data_start2 = (int16 __aie_dm_resource_b*)out_data.data();

     {
        static constexpr int16 DELTA = (LENGTH / (16*BATCH_SIZE));
        auto inDataIt = aie::begin_vector<BATCH_SIZE>(in_data_start);
        auto outDataIt = aie::begin_vector<BATCH_SIZE>(out_data_start);
        for (unsigned i=0; i < 4; i++) chess_prepare_for_pipelining {
            for (unsigned j=0; j < DELTA; j++) chess_unroll_loop(DELTA) {
                two_layer_ilv();
            }

            inDataIt += 3*DELTA;
            outDataIt += 3*DELTA;
        }
    }

    // Print for debug
#if DEBUGLEVEL == ALL
    {
        LOG_STATUS("AFTER FIRST TWO_ILV\n");
        auto outDataIt = aie::begin_vector<BATCH_SIZE>(out_data_start);
        for (unsigned i=0; i < LENGTH / BATCH_SIZE; i++) {
            aie::vector<int16, BATCH_SIZE> vec = *outDataIt++;
                LOG_VECTOR(vec, true, "\nvec=");
        }
    }
#endif


     {
        auto inDataIt = aie::begin_vector<BATCH_SIZE>(out_data_start);
        auto outDataIt = aie::begin_vector<BATCH_SIZE>(in_data_start);

        static constexpr int16 DELTA = (LENGTH / (4*BATCH_SIZE));
        for (unsigned j=0; j < DELTA; j++) { two_layer_ilv(); }
    }


    // Print for debug
#if DEBUGLEVEL == ALL
    {
        LOG_STATUS("AFTER SECOND TWO_ILV\n");
        auto outDataIt = aie::begin_vector<BATCH_SIZE>(in_data_start);
        for (unsigned i=0; i < LENGTH / BATCH_SIZE; i++) {
            aie::vector<int16, BATCH_SIZE> vec = *outDataIt++;
                LOG_VECTOR(vec, true, "\nvec=");
        }
    }
#endif
     
    {
        auto inDataIt = aie::begin_vector<BATCH_SIZE>(in_data_start);
        auto outDataIt = aie::begin_vector<BATCH_SIZE>(out_data_start);

        if constexpr (LENGTH == 256) {
            for (unsigned j=0; j < 16; j++) chess_unroll_loop(16) { vmd(0); } // bit-reversal of 0 bits
        } else if constexpr (LENGTH == 512) {
            for (unsigned j=0; j < 16; j++) { vmd(0);vmd(0); } // bit-reversal of 1 bit
        } else if constexpr (LENGTH == 1024) {
            for (unsigned j=0; j < 16; j++) { vmd(0);vmd(1);vmd(-1);vmd(0); } // bit-reversal of 2 bits
        } else if constexpr (LENGTH == 2048) {
            for (unsigned j=0; j < 16; j++) { vmd(0);vmd(3);vmd(0);vmd(3);vmd(-3);vmd(0);vmd(-3);vmd(0); } // bit-reversal of 3 bits
        } else if constexpr (LENGTH == 4096) {
            for (unsigned j=0; j < 16; j++) { vmd(0);vmd(7);vmd(2);vmd(9);vmd(-2);vmd(5);vmd(0);vmd(7);vmd(-7);vmd(0);vmd(-5);vmd(2);vmd(-9);vmd(-2);vmd(-7);vmd(0); } // bit-reversal of 8 bits
        }
    }


    // Print for debug
#if DEBUGLEVEL == ALL
    {
        LOG_STATUS("AFTER LAST STEP\n");
        auto outDataIt = aie::begin_vector<BATCH_SIZE>(out_data_start);
        for (unsigned i=0; i < LENGTH / BATCH_SIZE; i++) {
            aie::vector<int16, BATCH_SIZE> vec = *outDataIt++;
                LOG_VECTOR(vec, true, "\nvec=");
        }
    }
#endif
}