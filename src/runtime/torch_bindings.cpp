#include <torch/extension.h>
#include <cstdint>

#include "tilegen/abi.h"
#include "tilegen/registry.h"
#include "tilegen/runtime.h"
#include "tilegen/trace.h"

namespace {

// A minimal CPU reference kernel proving the dispatch path end-to-end.
void ref_add_cpu(const tilegen::KernelArgs& args) {
    const float* a = static_cast<const float*>(args.ptrs[0]);
    const float* b = static_cast<const float*>(args.ptrs[1]);
    float* c = static_cast<float*>(args.ptrs[2]);
    const int64_t n = args.ints[0];
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

const std::string kRefAddHash = "c0ffee0000000000000000000000000000000001";

struct RegistryBootstrap {
    RegistryBootstrap() {
        tilegen::KernelDescriptor d;
        d.name = "tilegen_ref_add";
        d.fn = ref_add_cpu;
        tilegen::register_kernel(kRefAddHash, d);
    }
};
RegistryBootstrap g_bootstrap;

torch::Tensor ref_add(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have identical shapes");
    auto ca = a.contiguous();
    auto cb = b.contiguous();
    auto c = torch::empty_like(ca);

    tilegen::KernelArgs args;
    args.ptrs[0] = ca.data_ptr<float>();
    args.ptrs[1] = cb.data_ptr<float>();
    args.ptrs[2] = c.data_ptr<float>();
    args.num_ptrs = 3;
    args.ints[0] = static_cast<int64_t>(ca.numel());
    args.num_ints = 1;

    tilegen::profiler_begin("tilegen::ref_add");
    tilegen::dispatch(kRefAddHash, args);
    tilegen::profiler_end("tilegen::ref_add");
    return c;
}

std::string semantic_hash(const std::string& canonical_json) {
    return tilegen::semantic_hash(canonical_json);
}

int64_t kernel_count() { return static_cast<int64_t>(tilegen::kernel_count()); }

std::string version() { return "0.1.0"; }

TORCH_LIBRARY(tilegen, m) {
    m.def("ref_add(Tensor a, Tensor b) -> Tensor");
    m.def("semantic_hash(str canonical_json) -> str");
    m.def("kernel_count() -> int");
    m.def("version() -> str");
}

TORCH_LIBRARY_IMPL(tilegen, CPU, m) {
    m.impl("ref_add", ref_add);
}

// Non-tensor ops (no tensor arguments) need a composite fallback so they
// dispatch on every backend key, not just the CPU key.
TORCH_LIBRARY_IMPL(tilegen, CompositeImplicitAutograd, m) {
    m.impl("version", version);
    m.impl("semantic_hash", semantic_hash);
    m.impl("kernel_count", kernel_count);
}

// Module entry point.  The artifact is installed as ``tilegen/_C.so`` (no
// platform tag), so the init symbol must be ``PyInit__C`` to match the module
// name ``tilegen._C``.  The ``TORCH_LIBRARY`` registrations above run as
// static initializers when the extension is imported.
PYBIND11_MODULE(_C, m) {
    m.doc() = "TileGen zero-Python-overhead C++ runtime bindings";
}

}  // namespace
