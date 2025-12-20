# CNN Hardware Modules

This repository contains all of the Verilog modules designed to perform moderate-throughput inference using 1D convolutional neural networks on FPGA hardware. The design emphasizes **determinism**, **integer efficiency**, and **instruction-driven control** with a focus on integer-only computation and straightforward hardware mapping.

---

## Overview

### `conv_core.sv`
This is the primary controller for all convolution operations. It delegates memory address 
determination and dot product computation to subordinate modules, but retains the function of populating
vectors with actual data for consumption by the int_8_vector_core and subsequently places the output
into an appropriate buffer for use in the next layer  The main loop of the conv_core module is concerned
with the completion of a single convolution, including optional activation and pooling:
- Depthwise, Pointwise, and Fully Connected layers are handled first
- ReLU is applied in the int8_vector_core if applicable
- MaxPool and Global Average Pool follow any concolution that may be present in an instruction.
From the perspective of the conv_core, each of these operations are identical and it is principally just
packing and unpacking vectors to complete its work.

### `int_8_vector_core.sv`
A specialized computation unit that:
- Computes, quantizes and applies activation to signed int8 vectors
- Handles groups of smaller vectors in parallel, up to a total count of 256 elements in power of two packages
- Is capable of processing and returning dot products of any compatible width in two cycles once the pipeline is filled
- The pipeline can contain a maximum of 7 256 wide vectors at a given time.
- Performs ReLU activation, and quantization of returned values if requested.

### `instruction_handler.sv` 
A module concerned with parsing instructions and using this information to determine:
- The optimal vector packing strategy for the layer given the power of two packaging size constraint
- Addresses translation for use by the conv_core module
- Tensor traversal
- Making pertinent information about a particular instruction available to the modules in need of said information

### `input_tensor.sv`  
This is a stub at present, used to allow demonstration of the CNN in action on real hardware.  It's intended function
is to read serial sensor input and to assemble individual time slices into a tensor for consumption by conv_core.  In a
similar way to the main module, this one should be driven by instructions rather than anything being hard
coded with respect to sensor modality or preprocessing operations.

---

## Current Status

These modules are under active development. The focus has been on:
- Refactoring for clarity and modularity
- Ensuring compatibility across convolution types
- Preparing for integration with the dot product engine

Near-term goals:
- Conclude testing of primary convolution logic
- Finalize and test pooling logic

---

## Design Philosophy

- **Static Control Flow**: No runtime loops over dynamic bounds. All control is driven by precompiled instruction fields.
- **Memory Locality**: Instructions and data are loaded from BRAM; no external memory or cache needed.
- **Predictable Timing**: Vectorization and traversal logic aim to make timing and performance analysis transparent.
- **Hardware-Generalized Convolution**: A single module interprets multiple layer types with consistent logic.
- **Single Producer and Consumer**: No multiplexing is required and there is no resource contention.

---

## Usage

Coming soon — an example testbench and minimal instructions to simulate the convolution controller with packed input tensors and kernel parameters.

---

## Related Work

This hardware is part of a broader exploration of integer only learning systems for embedded AI.  See the other sections of this repository for details on developing, training and exporting compatible models for this framework.
