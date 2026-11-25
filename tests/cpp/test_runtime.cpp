// Standalone runtime self-test (no torch dependency).
//
// Validates the SHA-256 implementation against FIPS known-answer vectors,
// exercises the kernel registry + dispatcher, and writes a JSON trace.
//
// Uses an explicit CHECK macro (not <cassert>) so checks run in every build
// configuration, including Release builds where NDEBUG disables assert().
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "tilegen/registry.h"
#include "tilegen/runtime.h"
#include "tilegen/trace.h"

using namespace tilegen;

#define CHECK(cond)                                            \
  do {                                                         \
    if (!(cond)) {                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                \
    }                                                          \
  } while (0)

namespace {

void ref_add(const KernelArgs& args) {
    const float* a = static_cast<const float*>(args.ptrs[0]);
    const float* b = static_cast<const float*>(args.ptrs[1]);
    float* c = static_cast<float*>(args.ptrs[2]);
    const int64_t n = args.ints[0];
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

}  // namespace

int main() {
    // sha256("") and sha256("abc"), first 40 hex chars.
    CHECK(semantic_hash("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4");
    CHECK(semantic_hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a3");
    {
        uint8_t digest[32];
        sha256(reinterpret_cast<const uint8_t*>(""), 0, digest);
        CHECK(digest[0] == 0xe3 && digest[1] == 0xb0 && digest[2] == 0xc4 &&
              digest[3] == 0x42);
    }

    const std::string key = "1111111111111111111111111111111111111111";
    KernelDescriptor desc;
    desc.name = "ref_add";
    desc.fn = ref_add;
    desc.grid_x = 1;
    desc.block_x = 128;
    CHECK(register_kernel(key, desc));
    CHECK(kernel_count() == 1);
    CHECK(!register_kernel(key, desc));  // duplicate rejected

    float a[4] = {1.f, 2.f, 3.f, 4.f};
    float b[4] = {10.f, 20.f, 30.f, 40.f};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    KernelArgs args;
    args.ptrs[0] = a;
    args.ptrs[1] = b;
    args.ptrs[2] = c;
    args.num_ptrs = 3;
    args.ints[0] = 4;
    args.num_ints = 1;

    CHECK(lookup_kernel(key) != nullptr);
    CHECK(dispatch(key, args));
    CHECK(c[0] == 11.f && c[3] == 44.f);
    CHECK(!dispatch("0000000000000000000000000000000000000000", args));

    // The dispatch above recorded a "ref_add" span; flush it to a file.
    const std::string trace_path = "/tmp/tilegen_runtime_trace.json";
    CHECK(profiler_flush(trace_path));
    {
        std::ifstream f(trace_path);
        CHECK(f.is_open());
        std::string body((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        CHECK(body.find("\"ref_add\"") != std::string::npos);
        CHECK(body.find("\"duration_us\"") != std::string::npos);
    }
    std::remove(trace_path.c_str());

    std::printf("runtime self-test passed\n");
    return 0;
}
