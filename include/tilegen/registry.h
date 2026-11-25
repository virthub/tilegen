// Public C/C++ ABI: kernel registry & dispatcher.
#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tilegen/abi.h"

namespace tilegen {

// Registers a kernel under a semantic-hash key.  Returns false if the key is
// already present (first registration wins).
bool register_kernel(const std::string& semantic_id, const KernelDescriptor& desc);

// Look up a registered kernel; returns nullptr when absent.
const KernelDescriptor* lookup_kernel(const std::string& semantic_id);

// Dispatch a kernel by semantic id.  Returns false if no kernel is registered.
bool dispatch(const std::string& semantic_id, const KernelArgs& args);

// Number of registered kernels (for tests / diagnostics).
size_t kernel_count();

}  // namespace tilegen
