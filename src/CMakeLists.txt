# ---------------------------------------------------------------------------
#  The automatic differentiation library.
#
#  It starts life header-only (an INTERFACE target).  The moment you add a
#  translation unit (*.cpp / *.cc / *.cxx) anywhere under src/, this switches
#  automatically to a compiled static library -- no edits required here.
# ---------------------------------------------------------------------------

file(GLOB_RECURSE AUTODIFF_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hh"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

file(GLOB_RECURSE AUTODIFF_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
)

# Headers live in src/autodiff/ but are included bare (<dual_number.h>), so the
# include root is that directory rather than src/.
if(AUTODIFF_SOURCES)
  add_library(autodiff STATIC ${AUTODIFF_SOURCES} ${AUTODIFF_HEADERS})
  target_include_directories(autodiff PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/autodiff")
  target_compile_features(autodiff PUBLIC cxx_std_26)
  target_link_libraries(autodiff PRIVATE autodiff::warnings)
  message(STATUS "autodiff: compiled static library (${CMAKE_CURRENT_SOURCE_DIR})")
else()
  add_library(autodiff INTERFACE)
  target_include_directories(autodiff INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/autodiff")
  target_compile_features(autodiff INTERFACE cxx_std_26)
  message(STATUS "autodiff: header-only (INTERFACE) library")
endif()

# ---------------------------------------------------------------------------
#  Optional MLX backend. When enabled, link it into the core library and define
#  AUTODIFF_WITH_MLX so the MLX-only headers (src/autodiff/mlx_*.h) can guard
#  their contents. `mlx` requires C++20; our C++26 setting already covers that.
# ---------------------------------------------------------------------------
if(AUTODIFF_MLX_ENABLED)
  get_target_property(_autodiff_type autodiff TYPE)
  if(_autodiff_type STREQUAL "INTERFACE_LIBRARY")
    set(_autodiff_scope INTERFACE)
  else()
    set(_autodiff_scope PUBLIC)
  endif()
  target_link_libraries(autodiff ${_autodiff_scope} mlx)
  target_compile_definitions(autodiff ${_autodiff_scope} AUTODIFF_WITH_MLX)
  message(STATUS "autodiff: linked against MLX ${MLX_VERSION}")
endif()

# Keep the headers visible in IDE project trees regardless of library kind.
if(AUTODIFF_HEADERS)
  add_custom_target(autodiff_headers SOURCES ${AUTODIFF_HEADERS})
endif()

add_library(autodiff::autodiff ALIAS autodiff)
