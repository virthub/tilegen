cmake_minimum_required(VERSION 3.18)
project(tilegen_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TILEGEN_BUILD_TESTS "Build the standalone C++ runtime self-test" ON)
option(TILEGEN_BUILD_TORCH_EXT "Build the PyTorch extension (tilegen._C)" OFF)

# ---------------------------------------------------------------------------
# Core zero-dependency runtime library
# ---------------------------------------------------------------------------
add_library(tilegen_runtime
  src/runtime/sha256.cpp
  src/runtime/trace.cpp
  src/runtime/dispatcher.cpp
  src/runtime/loader.cpp
)
target_include_directories(tilegen_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(tilegen_runtime PUBLIC ${CMAKE_DL_LIBS})
endif()

# ---------------------------------------------------------------------------
# Optional PyTorch extension
# ---------------------------------------------------------------------------
if(TILEGEN_BUILD_TORCH_EXT)
  find_package(Torch QUIET)
  if(Torch_FOUND)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    add_library(_C MODULE src/runtime/torch_bindings.cpp)
    set_target_properties(_C PROPERTIES
      PREFIX ""
      OUTPUT_NAME "_C"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/tilegen"
    )
    target_link_libraries(_C PRIVATE tilegen_runtime ${TORCH_LIBRARIES})
    target_include_directories(_C PRIVATE ${TORCH_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
    target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)
    set_target_properties(_C PROPERTIES CXX_VISIBILITY_PRESET default)
  else()
    message(WARNING "TILEGEN_BUILD_TORCH_EXT=ON but Torch was not found; skipping tilegen._C")
  endif()
endif()

# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------
if(TILEGEN_BUILD_TESTS)
  enable_testing()
  add_executable(tilegen_runtime_test tests/cpp/test_runtime.cpp)
  target_link_libraries(tilegen_runtime_test PRIVATE tilegen_runtime)
  add_test(NAME runtime_test COMMAND tilegen_runtime_test)
endif()

# ---------------------------------------------------------------------------
# Install rules (used by scikit-build-core / system installs)
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS tilegen_runtime
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/tilegen DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(TARGET _C)
  install(TARGETS _C LIBRARY DESTINATION tilegen)
endif()
