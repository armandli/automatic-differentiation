cmake_minimum_required(VERSION 3.25)

project(automatic_differentiation
  VERSION 0.1.0
  DESCRIPTION "A hobby automatic differentiation framework written in C++"
  HOMEPAGE_URL "https://github.com/armandli/automatic-differentiation"
  LANGUAGES CXX
)

# ---------------------------------------------------------------------------
#  Basic guards and global settings
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed. Run:\n"
    "  cmake -S . -B build\n"
    "or just use the provided Makefile (make / make test / make clean).")
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pick a default build type for single-config generators.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT CMAKE_BUILD_TYPE AND NOT _is_multi_config)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build configuration" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
#  Output layout: every built executable lands directly in <build>/
#  (static/shared libraries and CMake's own junk are tucked under <build>/lib)
# ---------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------------------------------------------------------
#  Options
# ---------------------------------------------------------------------------
option(AUTODIFF_BUILD_TESTS "Build the unit test suite" ON)
option(AUTODIFF_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(AUTODIFF_WITH_MLX "AUTO" CACHE STRING
  "MLX-backed backend: AUTO (enable if MLX is found), ON (required), or OFF")
set_property(CACHE AUTODIFF_WITH_MLX PROPERTY STRINGS AUTO ON OFF)

# ---------------------------------------------------------------------------
#  A shared "how we want our own code compiled" target.
#  Only applied to first-party targets, never to fetched dependencies.
# ---------------------------------------------------------------------------
add_library(autodiff_warnings INTERFACE)
add_library(autodiff::warnings ALIAS autodiff_warnings)
if(MSVC)
  target_compile_options(autodiff_warnings INTERFACE /W4 /permissive-)
  if(AUTODIFF_WARNINGS_AS_ERRORS)
    target_compile_options(autodiff_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(autodiff_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow)
  if(AUTODIFF_WARNINGS_AS_ERRORS)
    target_compile_options(autodiff_warnings INTERFACE -Werror)
  endif()
endif()

# ---------------------------------------------------------------------------
#  Optional dependencies
#
#  MLX (https://github.com/ml-explore/mlx) is Apple's array framework. It is
#  consumed as an installed package -- `brew install mlx`, or `pip install mlx`
#  then point CMake at it with
#    -DCMAKE_PREFIX_PATH=$(python3 -m mlx --cmake-dir)
#  The `mlx` imported target is promoted to GLOBAL so src/ and test/ can see it.
# ---------------------------------------------------------------------------
if(AUTODIFF_WITH_MLX STREQUAL "AUTO")
  find_package(MLX CONFIG QUIET GLOBAL)
elseif(AUTODIFF_WITH_MLX)
  find_package(MLX CONFIG REQUIRED GLOBAL)
endif()

set(AUTODIFF_MLX_ENABLED FALSE)
if(MLX_FOUND)
  set(AUTODIFF_MLX_ENABLED TRUE)
endif()

# ---------------------------------------------------------------------------
#  Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(AUTODIFF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# ---------------------------------------------------------------------------
#  Configuration summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "  ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  ----------------------------------------")
message(STATUS "  C++ standard .......... ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type ........... ${CMAKE_BUILD_TYPE}")
message(STATUS "  Generator ............ ${CMAKE_GENERATOR}")
message(STATUS "  Build tests .......... ${AUTODIFF_BUILD_TESTS}")
message(STATUS "  Warnings as errors ... ${AUTODIFF_WARNINGS_AS_ERRORS}")
if(AUTODIFF_MLX_ENABLED)
  message(STATUS "  MLX backend ......... ${AUTODIFF_MLX_ENABLED} (MLX ${MLX_VERSION}, Metal ${MLX_BUILD_METAL})")
else()
  message(STATUS "  MLX backend ......... ${AUTODIFF_MLX_ENABLED} (AUTODIFF_WITH_MLX=${AUTODIFF_WITH_MLX})")
endif()
message(STATUS "  Executable output .... ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")
