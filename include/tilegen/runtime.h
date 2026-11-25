// Public C/C++ ABI: semantic hash (SHA-256 over canonical JSON).
#pragma once

#include <cstdint>
#include <string>

namespace tilegen {

// SHA-256 digest as 64 lowercase hex characters.
//
// The Python layer (`tilegen.runtime.semantic_hash`) produces the identical
// digest over the identical canonical JSON string, so artifact cache keys
// computed at compile time match lookup keys computed at dispatch time.
std::string semantic_hash(const std::string& canonical_json);

// Raw SHA-256 over an arbitrary byte buffer (for tests / verification).
void sha256(const uint8_t* data, size_t len, uint8_t out[32]);

}  // namespace tilegen
