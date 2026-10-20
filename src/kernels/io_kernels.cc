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
 * Input/output kernels of the test harness (see DUT_GRAPH in project.cpp).
 *
 * The polynomial of LENGTH coefficients enters the AIE array through four PLIOs, each carrying one
 * quarter of the coefficients (q0 = coefficients [0, LENGTH/4), q1 = [LENGTH/4, LENGTH/2), ...).
 * The input kernel concatenates the four quarters into one buffer, the output kernel splits the
 * result buffer into four quarters again. These kernels only copy data; they are not part of the
 * NTT itself and are not included in the reported cycle counts.
 */


template<unsigned LENGTH>
void INPUT_KERNEL_INT16(adf::input_buffer<int16> & __restrict in_data_q0, adf::input_buffer<int16> & __restrict in_data_q1, adf::input_buffer<int16> & __restrict in_data_q2, adf::input_buffer<int16> & __restrict in_data_q3, adf::output_buffer<int16>& __restrict out_data) {

    static constexpr unsigned VECTOR_SIZE = 16;

    int16 __aie_dm_resource_a* __restrict in_buffer_q0 = (int16 __aie_dm_resource_a* __restrict) in_data_q0.data();
    auto inQ0It = aie::begin_vector<VECTOR_SIZE>(in_buffer_q0);

    int16 __aie_dm_resource_a* __restrict in_buffer_q1 = (int16 __aie_dm_resource_a* __restrict) in_data_q1.data();
    auto inQ1It = aie::begin_vector<VECTOR_SIZE>(in_buffer_q1);

    int16 __aie_dm_resource_a* __restrict in_buffer_q2 = (int16 __aie_dm_resource_a* __restrict) in_data_q2.data();
    auto inQ2It = aie::begin_vector<VECTOR_SIZE>(in_buffer_q2);

    int16 __aie_dm_resource_a* __restrict in_buffer_q3 = (int16 __aie_dm_resource_a* __restrict) in_data_q3.data();
    auto inQ3It = aie::begin_vector<VECTOR_SIZE>(in_buffer_q3);


    int16 __aie_dm_resource_b* __restrict out_buffer = (int16 __aie_dm_resource_b* __restrict) out_data.data();
    auto outIt = aie::begin_vector<VECTOR_SIZE>(out_buffer);

    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outIt++ = *inQ0It++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outIt++ = *inQ1It++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outIt++ = *inQ2It++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outIt++ = *inQ3It++; }

}


template<unsigned LENGTH>
void OUTPUT_KERNEL_INT16(adf::input_buffer<int16>& __restrict in_data, adf::output_buffer<int16> & __restrict out_data_q0, adf::output_buffer<int16> & __restrict out_data_q1, adf::output_buffer<int16> & __restrict out_data_q2, adf::output_buffer<int16> & __restrict out_data_q3) {

    static constexpr unsigned VECTOR_SIZE = 16;

    int16 __aie_dm_resource_a* __restrict out_buffer_q0 = (int16 __aie_dm_resource_a* __restrict) out_data_q0.data();
    auto outQ0It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q0);

    int16 __aie_dm_resource_a* __restrict out_buffer_q1 = (int16 __aie_dm_resource_a* __restrict) out_data_q1.data();
    auto outQ1It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q1);

    int16 __aie_dm_resource_a* __restrict out_buffer_q2 = (int16 __aie_dm_resource_a* __restrict) out_data_q2.data();
    auto outQ2It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q2);

    int16 __aie_dm_resource_a* __restrict out_buffer_q3 = (int16 __aie_dm_resource_a* __restrict) out_data_q3.data();
    auto outQ3It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q3);


    int16 __aie_dm_resource_b* __restrict in_buffer = (int16 __aie_dm_resource_b* __restrict) in_data.data();
    auto inIt = aie::begin_vector<VECTOR_SIZE>(in_buffer);


    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outQ0It++ = *inIt++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outQ1It++ = *inIt++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outQ2It++ = *inIt++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outQ3It++ = *inIt++; }

}



template<unsigned LENGTH>
void INPUT_KERNEL_INT32(adf::input_buffer<int32> & __restrict in_data_q0, adf::input_buffer<int32> & __restrict in_data_q1, adf::input_buffer<int32> & __restrict in_data_q2, adf::input_buffer<int32> & __restrict in_data_q3, adf::output_buffer<int32>& __restrict out_data) {

    static constexpr unsigned VECTOR_SIZE = 8;

    int32 __aie_dm_resource_a* __restrict in_buffer_q0 = (int32 __aie_dm_resource_a* __restrict) in_data_q0.data();
    auto inQ0It = aie::begin_vector<VECTOR_SIZE>(in_buffer_q0);

    int32 __aie_dm_resource_a* __restrict in_buffer_q1 = (int32 __aie_dm_resource_a* __restrict) in_data_q1.data();
    auto inQ1It = aie::begin_vector<VECTOR_SIZE>(in_buffer_q1);

    int32 __aie_dm_resource_a* __restrict in_buffer_q2 = (int32 __aie_dm_resource_a* __restrict) in_data_q2.data();
    auto inQ2It = aie::begin_vector<VECTOR_SIZE>(in_buffer_q2);

    int32 __aie_dm_resource_a* __restrict in_buffer_q3 = (int32 __aie_dm_resource_a* __restrict) in_data_q3.data();
    auto inQ3It = aie::begin_vector<VECTOR_SIZE>(in_buffer_q3);


    int32 __aie_dm_resource_b* __restrict out_buffer = (int32 __aie_dm_resource_b* __restrict) out_data.data();
    auto outIt = aie::begin_vector<VECTOR_SIZE>(out_buffer);


    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outIt++ = *inQ0It++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outIt++ = *inQ1It++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outIt++ = *inQ2It++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outIt++ = *inQ3It++; }

}


template<unsigned LENGTH>
void OUTPUT_KERNEL_INT32(adf::input_buffer<int32>& __restrict in_data, adf::output_buffer<int32> & __restrict out_data_q0, adf::output_buffer<int32> & __restrict out_data_q1, adf::output_buffer<int32> & __restrict out_data_q2, adf::output_buffer<int32> & __restrict out_data_q3) {

    static constexpr unsigned VECTOR_SIZE = 8;

    int32 __aie_dm_resource_a* __restrict out_buffer_q0 = (int32 __aie_dm_resource_a* __restrict) out_data_q0.data();
    auto outQ0It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q0);

    int32 __aie_dm_resource_a* __restrict out_buffer_q1 = (int32 __aie_dm_resource_a* __restrict) out_data_q1.data();
    auto outQ1It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q1);

    int32 __aie_dm_resource_a* __restrict out_buffer_q2 = (int32 __aie_dm_resource_a* __restrict) out_data_q2.data();
    auto outQ2It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q2);

    int32 __aie_dm_resource_a* __restrict out_buffer_q3 = (int32 __aie_dm_resource_a* __restrict) out_data_q3.data();
    auto outQ3It = aie::begin_vector<VECTOR_SIZE>(out_buffer_q3);


    int32 __aie_dm_resource_b* __restrict in_buffer = (int32 __aie_dm_resource_b* __restrict) in_data.data();
    auto inIt = aie::begin_vector<VECTOR_SIZE>(in_buffer);


    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outQ0It++ = *inIt++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outQ1It++ = *inIt++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outQ2It++ = *inIt++; }
    for (int i = 0; i < LENGTH / (4*VECTOR_SIZE); i++) { *outQ3It++ = *inIt++; }

}
