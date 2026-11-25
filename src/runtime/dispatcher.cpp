#include "tilegen/registry.h"
#include "tilegen/trace.h"

#include <mutex>
#include <unordered_map>

namespace tilegen {
namespace {

std::mutex& registry_mutex() {
    static std::mutex mu;
    return mu;
}

using Registry = std::unordered_map<std::string, KernelDescriptor>;

Registry& registry() {
    static Registry* reg = new Registry();
    return *reg;
}

}  // namespace

bool register_kernel(const std::string& semantic_id, const KernelDescriptor& desc) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    if (desc.fn == nullptr) return false;
    return registry().emplace(semantic_id, desc).second;
}

const KernelDescriptor* lookup_kernel(const std::string& semantic_id) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto it = registry().find(semantic_id);
    return it == registry().end() ? nullptr : &it->second;
}

bool dispatch(const std::string& semantic_id, const KernelArgs& args) {
    const KernelDescriptor* desc = lookup_kernel(semantic_id);
    if (desc == nullptr) return false;
    profiler_begin(desc->name);
    desc->fn(args);
    profiler_end(desc->name);
    return true;
}

size_t kernel_count() {
    std::lock_guard<std::mutex> lk(registry_mutex());
    return registry().size();
}

}  // namespace tilegen
