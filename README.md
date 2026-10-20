# NTTs on AMD AI Engines (AIE1)

Artifact for the paper

> **Software-Pipelining The Number Theoretic Transform on Neural Processing Units: A Loop-Fission Methodology**
> Henri Vermeersch (1,2), Thomas de Ruijter (1), Quinten Norga (1), Xander Pottier (1) and Ingrid Verbauwhede (1,2)
> IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)
>
> (1) COSIC, KU Leuven, Leuven, Belgium, `{firstname.lastname}@kuleuven.be`
> (2) Belfort Labs, Leuven, Belgium, `{firstname.lastname}@belfortlabs.com`

Contact for the artifact: Henri Vermeersch (henri.vermeersch@kuleuven.be).

This repository contains the AI Engine source code of the number theoretic transform (NTT)
implementations described in the paper, for AMD's first-generation AI Engine (AIE1) in the Versal
VCK190: the radix-2, radix-4 and radix-8 NTT pipelines (`int16` and `int32`), the Barrett reduction
primitives they are built from, and the vector-based bit-reversal unit. Every design can be compiled
with AMD Vitis 2024.1 and simulated with the cycle-accurate AI Engine simulator (`aiesimulator`), which
produced the results in the paper, or with the functional x86 simulator (`x86simulator`).

The artifact does not include the host or PL code for running the designs on a physical VCK190 board
(the "P" measurement in Section 5.4 of the paper); all reported cycle counts can be reproduced in
simulation.

## Contents

| Paper | Description | Code |
|---|---|---|
| Section 3 | Loop fission methodology: every kernel is split into small software-pipelined loops | all files in `src/kernels/` |
| Section 4, Figure 3, Table 1 | Partial Barrett multiplication/reduction, `int16` | `barrett_multiplication`, `barrett_reduction` in `src/kernels/ntt_r2_kernels.cc` |
| Section 4, Figure 3, Table 1 | Partial Barrett multiplication/reduction, `int32` (two loops) | `ntt_int32_*` helpers in `src/kernels/ntt_r2_int32_kernels.cc` |
| Section 4.2, Table 1 | Full Barrett multiplication by a constant, `int16` | `src/kernels/full_reduction_kernels.cc` |
| Section 5.1 | Radix-2 NTT, `int16`, forward and inverse | `src/ntt.h`, `src/kernels/ntt_r2_kernels.cc` |
| Section 5.1 | Radix-2 NTT, `int32`, forward and inverse | `src/ntt_int32.h`, `src/kernels/ntt_r2_int32_kernels.cc` |
| Section 5.2 | Radix-4 NTT, `int16`, forward | `src/ntt_r4.h`, `src/kernels/ntt_r4_kernels.cc` |
| Section 5.3 | Radix-8 NTT, `int16` (one pipeline stage = 2 tiles) | `src/ntt_r8.h`, `src/kernels/ntt_r8_kernels.cc` |
| Section 6, Table 6 | Vector-based bit-reversal unit, `int16` | `src/kernels/bitreverse_kernels.cc` |


## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| AMD Vitis (unified software platform, includes the AI Engine tools) | **2024.1** | provides `v++`, `aiesimulator`, `x86simulator`, `vitis_analyzer`; the AI Engine tools need a valid AMD license |
| VCK190 base platform | `xilinx_vck190_base_202410_1` | ships with Vitis 2024.1 in `$XILINX_VITIS/base_platforms/` |
| Operating system | a Linux distribution supported by Vitis 2024.1 (e.g. Ubuntu 20.04/22.04, RHEL 8) | |

No FPGA board is needed.

The code uses the AI Engine API and the AIE1 intrinsics (e.g. `mul16`, `mac16`, `lmul8`, `select32`),
so it only targets first-generation AI Engines (Versal AI Core, e.g. VCK190), not AIE-ML/AIE2.


## Building and running

The designs are built and simulated with the standard AMD Vitis 2024.1 AI Engine flow, either from
the command line (`v++ --mode aie`, `aiesimulator`, `x86simulator`) or in the Vitis Unified IDE, as
an *AI Engine component* for the `xilinx_vck190_base_202410_1` platform. For this artifact:

- the top-level graph file is `src/project.cpp`;
- `src/` must be on the include path (the kernel sources are referenced relative to it, e.g.
  `kernels/ntt_r2_kernels.cc`);
- the simulators read the input files relative to the repository root (`data/`, `data_int32/`);
- the design under test is selected with `NTT_CONFIG` (see "Configurations" below);
- for the cycle counts, use the AIE (hardware) target and the cycle-accurate `aiesimulator` with
  profiling enabled; the x86 target and `x86simulator` only check functionality.

### Vitis GUI (Vitis Unified IDE 2024.1)

The designs were developed in the Vitis GUI:

1. Start the IDE with `vitis -w <workspace directory>` (after sourcing `settings64.sh` of the Vitis
   installation).
2. Create a new *AI Engine component* (*File > New Component > AI Engine*) and select the
   `xilinx_vck190_base_202410_1` platform.
3. Import the `src/` folder (keeping the `src/kernels/` structure) and the `data/` and `data_int32/`
   folders into the component, and select `src/project.cpp` as the top-level graph file.
4. In the component's AI Engine compiler settings, add the component's `src` directory as an include
   directory.
5. Select the design under test by setting `NTT_CONFIG` at the top of `src/project.cpp`
   (see "Configurations" below).
6. In the *Flow* view, build and run for *AIE Simulator / Hardware* (cycle-accurate, with profiling
   enabled in the run settings) or for *X86 Simulation* (functional only).
7. Open the generated run summary in the *Analysis* view to inspect the profile and cycle counts
   (see "Interpreting the output").

### Command line and further documentation

The same flow can be run from the command line: the AMD tutorials below provide a command-line
variant (Makefile-based) of the GUI flow, and UG1076 documents the `v++ --mode aie`, `aiesimulator`
and `x86simulator` commands and their options.

Refer to AMD's official documentation for Vitis 2024.1 for the exact steps (select version 2024.1 on
docs.amd.com if a link opens a newer version):

- [AI Engine Tools and Flows User Guide (UG1076)](https://docs.amd.com/r/2024.1-English/ug1076-ai-engine-environment):
  compiling an AI Engine graph with `v++ --mode aie`, the AI Engine simulator (`aiesimulator`) and the
  x86 functional simulator (`x86simulator`), simulator options, and profiling.
- [Vitis Unified Software Platform Documentation (UG1393)](https://docs.amd.com/r/2024.1-English/ug1393-vitis-application-acceleration):
  installation and environment setup, the Vitis Unified IDE, and creating and building an AI Engine
  component in the GUI.
- [AI Engine Kernel and Graph Programming Guide (UG1079)](https://docs.amd.com/r/2024.1-English/ug1079-ai-engine-kernel-coding):
  the ADF graph and kernel programming model used in `src/` (kernel classes, buffers, PLIOs,
  location constraints).
- [Vitis Tutorials: AI Engine (XD100)](https://docs.amd.com/r/2024.1-English/Vitis-Tutorials-AI-Engine-Development)
  and the corresponding [GitHub repository (branch 2024.1)](https://github.com/Xilinx/Vitis-Tutorials/tree/2024.1/AI_Engine_Development/AIE):
  step-by-step examples of creating, compiling and simulating AI Engine designs, both in the Vitis
  Unified IDE and on the command line (see e.g. "Porting a Command Line Project to the Vitis IDE
  Project").


## Configurations

`src/project.cpp` instantiates exactly one design under test (DUT), selected by setting the constant
`NTT_CONFIG` at the top of `src/project.cpp` (default `0`).

| `NTT_CONFIG` | Design | n | q | AIE tiles (excl. I/O) | Paper |
|---|---|---|---|---|---|
| 0 (default) | Radix-2 `int16` forward NTT | 4096 | 12289 | 12 | Section 5.1, Tables 2, 3, 4 |
| 1 | Radix-2 `int16` inverse NTT | 4096 | 12289 | 12 | Section 5.1, Table 2 |
| 2 | Radix-4 `int16` forward NTT | 4096 | 12289 | 6 | Section 5.2, Tables 2, 3 |
| 3 | Radix-8 `int16`, one stage (butterfly tile + reduction tile) | 4096 | 12289 | 2 | Section 5.3, Table 2 |
| 4 | Radix-2 `int32` forward NTT | 2048 | 1073707009 | 11 | Section 5.1, Tables 2, 3, 4 |
| 5 | Radix-2 `int32` inverse NTT | 2048 | 1073707009 | 11 | Section 5.1 |
| 6 | Bit-reversal unit `int16` (size set by `BITREV_N`, default 4096) | 256 to 4096 | - | 1 | Section 6, Table 6 |
| 7 | Full Barrett multiplication `int16` by n^-1 mod q | 4096 | 12289 | 1 | Section 4.2, Table 1 |

Other compile-time options in `src/project.cpp`: `ITERATIONS` (number of transforms to simulate,
default 100) and `BITREV_N`. The moduli, roots of unity and the location of the pipeline in the AIE
array are also set at the top of `src/project.cpp`; the graph classes accept any NTT-friendly modulus
within the bounds of Table 3 of the paper. The debug output of the kernels is controlled by
`DEBUGLEVEL` in `src/kernels/debug.h` (`NO_PRINTS` for performance measurements; `ALL` prints
intermediate vectors in x86 simulation).

Every design is wrapped in the same test harness (`DUT_GRAPH` in `src/project.cpp`): the polynomial is
streamed in through four 128-bit PLIOs, each carrying one quarter of the coefficients
(`data/input_data_q{0..3}.txt` for `int16`, `data_int32/input_data_q{0..3}.txt` for `int32`); an input
kernel gathers the quarters into one buffer for the first tile of the pipeline, and an output kernel
splits the result over four output PLIOs. The input and output kernels are not part of the NTT and
are not included in the reported cycle counts. Since the kernels have no data-dependent control flow,
their cycle counts do not depend on the input values.


## Interpreting the output

### Functional output

The simulators write the transformed polynomials to
`data/output_data_q{0..3}.txt` in their output directory. Each file contains one
quarter of every output polynomial (in the order of the input quarters), with 8 (`int16`) or 4
(`int32`) coefficients per line. The cycle-accurate simulator additionally writes a timestamp
(`T <time> ns`) before every block of data. Note the following properties of the outputs:

- Coefficients are *partially reduced*: they are correct modulo q but lie in a range around zero
  (e.g. |a| <= 3q/4 after a Barrett multiplication, larger after the lazy-reduced butterflies). A full
  reduction to [0, q) is not part of the NTT pipelines (see Section 5.4 of the paper).
- The forward transforms take inputs in natural order and produce outputs in bit-reversed order; the
  inverse transforms take bit-reversed inputs and produce natural-order outputs. The inverse
  transforms do not include the multiplication by n^-1 (see "Full reduction and normalization factor"
  in Section 5.4); `NTT_CONFIG=7` implements that multiplication combined with a full reduction.

### Cycle counts (reproducing the paper)

Run the cycle-accurate `aiesimulator` with profiling enabled and `DEBUGLEVEL NO_PRINTS`, and open the
run summary (`default.aierun_summary` in the simulator output directory) in `vitis_analyzer` or in
the *Analysis* view of the Vitis IDE. In the *Profile* view, every NTT layer tile reports the cycles spent in its kernel function. The
cycles per kernel call (total cycles divided by the number of calls, `ITERATIONS`) of one layer tile is
the **throughput** reported in Table 2 (cycles per transform, the pipeline accepts a new polynomial every
such period), and the **latency** is this number multiplied by the number of pipeline stages. Since all
layers run the same kernel (constant-geometry transform), all layer tiles report about the same number;
for radix-8 the throughput is that of the slower of the two tiles. The time stamps in the output files
give the same throughput: in steady state, consecutive output polynomials are one throughput period
apart (1 cycle = 0.8 ns at 1.25 GHz).

The values reported in the paper (cycle-accurate simulation, 1.25 GHz):

| Design | `NTT_CONFIG` | Throughput (cycles) | Pipeline stages | Latency (cycles) | Paper |
|---|---|---|---|---|---|
| Radix-2 `int16` forward, n = 4096 | 0 | 952 | 12 | 11,424 | Table 2 |
| Radix-2 `int16` inverse, n = 4096 | 1 | 918 | 12 | 11,016 | Table 2 |
| Radix-4 `int16` | 2 | 808 (n = 2048) | 6 | 4,848 | Table 2 |
| Radix-8 `int16` | 3 | 770 (n = 2048) | 8 | 6,160 | Table 2 |
| Radix-2 `int32` forward, n = 2048 | 4 | 1,028 | 11 | 11,308 | Table 2 |
| Bit-reversal, n = 256 / 512 / 1024 / 2048 / 4096 | 6 | 100 / 156 / 268 / 492 / 940 | 1 | - | Table 6 |

Table 3 of the paper gives the corresponding *II approximation* (sum of the initiation intervals of all
loops of a kernel, without loop prologues/epilogues). The II of every software-pipelined loop can be
checked in the generated assembly in the compiler's work directory (`Work/aie/<col>_<row>/Release/`,
one directory per tile) and compared with the schedules in Figure 3 of the paper. For the Barrett primitives
(Table 1), the II per 16 (`int16`) or 8 (`int32`) lanes is the II of the corresponding loops in these
listings; the full Barrett multiplication (`NTT_CONFIG=7`) takes 1 + 2 + 1 = 4 cycles per 16
coefficients over its three loops.


## Source code organization

```
.
├── LICENSE                      MIT license
├── README.md                    this file
├── data/                        int16 input polynomials, one quarter of the coefficients per file
├── data_int32/                  int32 input polynomials, one quarter of the coefficients per file
└── src/
    ├── project.cpp              top-level graph: test harness (DUT_GRAPH) and selection of the design (NTT_CONFIG)
    ├── ntt.h                    radix-2 int16 pipeline (graph): one tile per layer, forward and inverse
    ├── ntt_int32.h              radix-2 int32 pipeline (graph)
    ├── ntt_r4.h                 radix-4 int16 pipeline (graph)
    ├── ntt_r8.h                 radix-8 int16 pipeline stage (graph)
    ├── standalone_units.h       graphs for the bit-reversal unit and the full Barrett multiplication
    ├── twiddle.h                graph-construction-time computation of twiddle factors and Barrett multipliers
    ├── auxiliary.h              compile-time helpers (log2, order of the root of unity, modular inverse, ...)
    └── kernels/
        ├── kernels.h            declarations of all kernels and kernel classes
        ├── debug.h              debug print macros (DEBUGLEVEL)
        ├── ntt_r2_kernels.cc    radix-2 int16 layer kernels + int16 partial Barrett primitives
        ├── ntt_r2_int32_kernels.cc  radix-2 int32 layer kernels + int32 partial Barrett primitives
        ├── ntt_r4_kernels.cc    radix-4 int16 layer kernel
        ├── ntt_r8_kernels.cc    radix-8 int16 butterfly and 32-to-16-bit reduction kernels
        ├── bitreverse_kernels.cc  vector-based bit-reversal unit
        ├── full_reduction_kernels.cc  full Barrett multiplication by a constant (int16)
        └── io_kernels.cc        input/output kernels of the test harness
```

The code follows the structure of Section 3.3 of the paper. The *graph* headers (`src/*.h`) build the
pipeline: one kernel per AIE tile, placed next to each other, with the twiddle factors and Barrett
multipliers of every layer computed when the graph is constructed and stored as kernel parameters in
the local memory of the tile. The *kernels* (`src/kernels/*.cc`) implement one NTT layer as a sequence
of small loops (manual loop fission), each written so that the AIE compiler can software-pipeline it at
its minimum II:

- `__aie_dm_resource_{a,b,c,d}` annotations tell the compiler which buffers are accessed in the same
  loop, so that they are placed in different memory banks (bank-disjointness invariant, Section 3.2);
- `chess_storage(...)` pins values to specific registers and some loops are unrolled manually to obtain
  the intended schedule (Section 3.3, step 4);
- the permutations (perfect shuffle, unshuffle, bit-reversal within a vector) are folded into the
  lane-selection arguments of the MAC intrinsics (`mul16`, `mac16`, `lmul8`), which multiply with
  constant vectors of {-1, 0, 1} entries (Section 3.3, step 3).


## License

This artifact is released under the MIT license, see `LICENSE`.
