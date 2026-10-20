
/*
 * Top-level AI Engine graph of the artifact for
 *   "Software-Pipelining The Number Theoretic Transform on Neural Processing Units: A Loop-Fission Methodology"
 *
 * Exactly one design under test (DUT) is instantiated, selected with NTT_CONFIG (see the README):
 *
 *   NTT_CONFIG  Design                                            Paper
 *   ----------  ------------------------------------------------  ------------------------------
 *   0           Radix-2 int16 forward NTT (CT),   n = 4096        Section 5.1, Tables 2-4 (default)
 *   1           Radix-2 int16 inverse NTT,        n = 4096        Section 5.1, Table 2
 *   2           Radix-4 int16 forward NTT,        n = 4096        Section 5.2, Tables 2-3
 *   3           Radix-8 int16, one pipeline stage, n = 4096       Section 5.3, Table 2
 *   4           Radix-2 int32 forward NTT (CT),   n = 2048        Section 5.1, Tables 2-4
 *   5           Radix-2 int32 inverse NTT (GS),   n = 2048        Section 5.1
 *   6           Bit-reversal unit int16,          n = BITREV_N    Section 6, Table 6
 *   7           Full Barrett multiplication int16 by n^-1 mod q   Section 4.2, Table 1
 *
 * Select a configuration by setting NTT_CONFIG below.
 */

#include "adf/new_frontend/adf.h"
#include "auxiliary.h"
#include "kernels/kernels.h"
#include "ntt.h"
#include "ntt_int32.h"
#include "ntt_r4.h"
#include "ntt_r8.h"
#include "standalone_units.h"

using namespace adf;


// Design under test, see the table above
#define NTT_CONFIG 0

// Polynomial size for the bit-reversal unit (NTT_CONFIG 6): 256, 512, 1024, 2048 or 4096
static constexpr unsigned BITREV_N = 4096;

// Number of graph iterations (= number of transforms) to simulate. The input files in data/ and
// data_int32/ contain enough coefficients for at least 150 iterations.
static constexpr unsigned ITERATIONS = 100;


// Parameters used in the paper
// int16: q = 12289, 41 is a primitive 4096-th root of unity modulo q
static constexpr uint16 Q_INT16 = 12289;
static constexpr uint16 ROOT_INT16 = 41;
// int32: q = 1073707009 (30 bits), 992735929 is a primitive 2048-th root of unity modulo q
static constexpr int32 Q_INT32 = 1073707009;
static constexpr int32 ROOT_INT32 = 992735929;
// Location (row, column) of the first tile of the pipeline in the AIE array
static constexpr unsigned STARTROW = 3;
static constexpr unsigned STARTCOL = 2;



/* DUT CLASS
 *
 * Test harness around a design T with an input port in_data and an output port out_data that both
 * carry one polynomial of LENGTH coefficients. The polynomial is streamed in through four 128-bit
 * PLIOs (one per quarter, files <input_folder>/input_data_q{0..3}.txt) and gathered into one buffer
 * by the input kernel; the output kernel splits the result into four quarters again, written to
 * data/output_data_q{0..3}.txt in the simulator output directory.
 */
template<unsigned LENGTH, typename T, bool INT32=false>
class DUT_GRAPH: public adf::graph {
    T dut;

public:
  kernel input_kernel;
  kernel output_kernel;

  input_plio in_data_Q0;
  input_plio in_data_Q1;
  input_plio in_data_Q2;
  input_plio in_data_Q3;

  output_plio out_data_Q0;
  output_plio out_data_Q1;
  output_plio out_data_Q2;
  output_plio out_data_Q3;

    DUT_GRAPH() {
        if constexpr (INT32) {
            input_kernel = kernel::create(INPUT_KERNEL_INT32<LENGTH>);
        } else {
            input_kernel = kernel::create(INPUT_KERNEL_INT16<LENGTH>);
        }
        source(input_kernel) = "kernels/io_kernels.cc";
        runtime<ratio>(input_kernel) = 1.0;
        dimensions(input_kernel.in[0]) = {LENGTH/4};
        dimensions(input_kernel.in[1]) = {LENGTH/4};
        dimensions(input_kernel.in[2]) = {LENGTH/4};
        dimensions(input_kernel.in[3]) = {LENGTH/4};
        dimensions(input_kernel.out[0]) = {LENGTH};

        if constexpr (INT32) {
            output_kernel = kernel::create(OUTPUT_KERNEL_INT32<LENGTH>);
        } else {
            output_kernel = kernel::create(OUTPUT_KERNEL_INT16<LENGTH>);
        }
        source(output_kernel) = "kernels/io_kernels.cc";
        runtime<ratio>(output_kernel) = 1.0;
        dimensions(output_kernel.out[0]) = {LENGTH/4};
        dimensions(output_kernel.out[1]) = {LENGTH/4};
        dimensions(output_kernel.out[2]) = {LENGTH/4};
        dimensions(output_kernel.out[3]) = {LENGTH/4};
        dimensions(output_kernel.in[0]) = {LENGTH};

        std::string input_folder = INT32 ? "data_int32" : "data";


        in_data_Q0  = input_plio::create(plio_128_bits, input_folder + "/input_data_q0.txt");
        in_data_Q1  = input_plio::create(plio_128_bits, input_folder + "/input_data_q1.txt");
        in_data_Q2  = input_plio::create(plio_128_bits, input_folder + "/input_data_q2.txt");
        in_data_Q3  = input_plio::create(plio_128_bits, input_folder + "/input_data_q3.txt");
        connect(in_data_Q0.out[0], input_kernel.in[0]);
        connect(in_data_Q1.out[0], input_kernel.in[1]);
        connect(in_data_Q2.out[0], input_kernel.in[2]);
        connect(in_data_Q3.out[0], input_kernel.in[3]);
        connect(input_kernel.out[0], dut.in_data);

        out_data_Q0 = output_plio::create(plio_128_bits, "data/output_data_q0.txt");
        out_data_Q1 = output_plio::create(plio_128_bits, "data/output_data_q1.txt");
        out_data_Q2 = output_plio::create(plio_128_bits, "data/output_data_q2.txt");
        out_data_Q3 = output_plio::create(plio_128_bits, "data/output_data_q3.txt");
        connect(dut.out_data, output_kernel.in[0]);
        connect(output_kernel.out[0], out_data_Q0.in[0]);
        connect(output_kernel.out[1], out_data_Q1.in[0]);
        connect(output_kernel.out[2], out_data_Q2.in[0]);
        connect(output_kernel.out[3], out_data_Q3.in[0]);
    }
};



/* Design under test */
#if NTT_CONFIG == 0
DUT_GRAPH<4096, NTT<4096, Q_INT16, ROOT_INT16, STARTROW, STARTCOL, NTT_TYPE::TYPE_NTT_NO_to_BO>> graph_dut;
#elif NTT_CONFIG == 1
DUT_GRAPH<4096, NTT<4096, Q_INT16, ROOT_INT16, STARTROW, STARTCOL, NTT_TYPE::TYPE_INTT_BO_to_NO>> graph_dut;
#elif NTT_CONFIG == 2
DUT_GRAPH<4096, NTT_R4<4096, Q_INT16, ROOT_INT16, STARTROW, STARTCOL, NTT_R4_TYPE::FORWARD_CT_NO_TO_BO>> graph_dut;
#elif NTT_CONFIG == 3
DUT_GRAPH<4096, NTT_R8<4096, Q_INT16, ROOT_INT16>> graph_dut;
#elif NTT_CONFIG == 4
DUT_GRAPH<2048, NTT_INT32<2048, Q_INT32, ROOT_INT32, STARTROW, STARTCOL, NTT_INT32_TYPE::FORWARD_CT_NO_TO_BO>, true> graph_dut;
#elif NTT_CONFIG == 5
DUT_GRAPH<2048, NTT_INT32<2048, Q_INT32, ROOT_INT32, STARTROW, STARTCOL, NTT_INT32_TYPE::INVERSE_GS_BO_TO_NO>, true> graph_dut;
#elif NTT_CONFIG == 6
DUT_GRAPH<BITREV_N, BITREVERSE<BITREV_N>> graph_dut;
#elif NTT_CONFIG == 7
DUT_GRAPH<4096, FULL_BARRETT_MULTIPLICATION<4096, Q_INT16, ExtendedGCD<4096, Q_INT16>::value>> graph_dut;
#else
#error "Unknown NTT_CONFIG, see the table at the top of project.cpp"
#endif



int main(void) {
  graph_dut.init();
  graph_dut.run(ITERATIONS);
  graph_dut.end();

  return 0;
}
