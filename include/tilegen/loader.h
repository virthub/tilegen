// Public C/C++ ABI: binary artifact loader (.cubin / .hsaco / .so).
#pragma once

#include <string>

#include "tilegen/abi.h"

namespace tilegen {

// Load a compiled artifact and fill ``desc`` with its entry point.
//
// Phase 1 supports dynamically-loadable shared objects exposing a
// ``extern "C" void tilegen_kernel(const tilegen::KernelArgs&)`` symbol; the
// CUDA/ROCm/Ascend binary loaders land with their respective backends in
// later phases.  Returns true on success.
bool load_binary(const std::string& path, KernelDescriptor* desc);

}  // namespace tilegen
