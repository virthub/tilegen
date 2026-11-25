// Public C/C++ ABI: runtime kernel descriptor & launch arguments.
//
// This header is the contract between the Python layer (semantic-hash
// computation, artifact cache) and the zero-Python-overhead C++ dispatcher.
#pragma once

#include <cstdint>
#include <cstddef>

namespace tilegen {

// Maximum number of I/O tensor pointers + scalar args passed in a single
// dispatch.  Keep small so the hot path stays on the stack.
constexpr int kMaxKernelArgs = 16;

// A device (or host) kernel argument bundle.  Quantized operands pass their
// packed data pointer, scales, and (optional) zero points explicitly.
struct KernelArgs {
    // Data pointers (global / scratchpad payloads).  Input tensors are
    // logically const; outputs are written through the same array.
    void* ptrs[kMaxKernelArgs];
    int num_ptrs = 0;

    // Scalar parameters (grid-relevant dims, strides, scale factors).
    int64_t ints[kMaxKernelArgs];
    int num_ints = 0;

    float floats[kMaxKernelArgs];
    int num_floats = 0;
};

// A registered kernel entry point.  ``fn`` is the compiled body; ``name`` is
// for trace/profiling output.
struct KernelDescriptor {
    const char* name = nullptr;
    void (*fn)(const KernelArgs&) = nullptr;
    // Static launch metadata (grid/block) — informational in the CPU reference
    // runtime, authoritative for the GPU launcher in later phases.
    uint32_t grid_x = 1;
    uint32_t grid_y = 1;
    uint32_t grid_z = 1;
    uint32_t block_x = 1;
    uint32_t block_y = 1;
    uint32_t block_z = 1;
};

}  // namespace tilegen
