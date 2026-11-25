#include "tilegen/loader.h"

#include <dlfcn.h>

namespace tilegen {

using KernelFn = void (*)(const KernelArgs&);

bool load_binary(const std::string& path, KernelDescriptor* desc) {
    if (desc == nullptr) return false;
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) return false;

    // Clear any prior dlopen error state.
    dlerror();
    void* sym = dlsym(handle, "tilegen_kernel");
    const char* err = dlerror();
    if (err != nullptr || sym == nullptr) {
        dlclose(handle);
        return false;
    }

    desc->name = "loaded";
    desc->fn = reinterpret_cast<KernelFn>(sym);
    return true;
}

}  // namespace tilegen
