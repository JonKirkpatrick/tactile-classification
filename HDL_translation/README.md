# CNN Hardware Modules

This repository contains two tightly coupled Verilog modules designed to perform high-throughput inference using 1D convolutional neural networks on FPGA hardware. The design emphasizes **determinism**, **integer efficiency**, and **instruction-driven control** with a focus on integer-only computation and straightforward hardware mapping.

---

## Overview

### `convolution.sv`
This is the primary controller for all convolution operations. It decodes a compact 64-bit instruction set and handles:
- Depthwise, pointwise, and fully connected layers
- Padding and stride
- Temporal traversal
- Vector preparation for dot product
- ReLU activation and (soon) MaxPooling

### `int_8_vector_core.sv`
A specialized computation unit that:
- Computes, quantizes and applies activation to signed int8 vectors
- Handles groups of smaller vectors in parallel, up to a total count of 256 elements in power of two packages
- Will process and return dot products of any compatible width in two cycles once the pipeline is filled
- The pipeline can contain a maximum of 7 256 wide vectors at a given time.

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

---

## Usage

Coming soon — an example testbench and minimal instructions to simulate the convolution controller with packed input tensors and kernel parameters.

---

## Related Work

This hardware is part of a broader exploration of fixed-point learning systems for embedded AI. See the forthcoming technical note *Learning in Fast Forward* for architectural details.
