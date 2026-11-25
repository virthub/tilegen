// Public C/C++ ABI: low-overhead JSON trace profiling.
//
// The dispatcher records structured timing events; when the
// ``TILEGEN_PROFILE_OUTPUT`` environment variable is set (or
// ``profiler_flush`` is called explicitly) the events are written as JSON.
#pragma once

#include <string>

namespace tilegen {

// Begin a named span.  Nesting is not required; unmatched begin/end pairs are
// ignored on the matching side.
void profiler_begin(const std::string& name);

// End the most recently begun span with the given name (if it matches).
void profiler_end(const std::string& name);

// Write accumulated events to ``path`` as JSON.  If ``path`` is empty, the
// ``TILEGEN_PROFILE_OUTPUT`` environment variable is consulted; if that is
// unset, nothing is written.  Returns true if a file was written.
bool profiler_flush(const std::string& path = "");

// Drop all accumulated events without writing.
void profiler_reset();

}  // namespace tilegen
