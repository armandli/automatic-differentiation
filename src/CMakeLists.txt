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

if(AUTODIFF_SOURCES)
  add_library(autodiff STATIC ${AUTODIFF_SOURCES} ${AUTODIFF_HEADERS})
  target_include_directories(autodiff PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(autodiff PUBLIC cxx_std_26)
  target_link_libraries(autodiff PRIVATE autodiff::warnings)
  message(STATUS "autodiff: compiled static library (${CMAKE_CURRENT_SOURCE_DIR})")
else()
  add_library(autodiff INTERFACE)
  target_include_directories(autodiff INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(autodiff INTERFACE cxx_std_26)
  message(STATUS "autodiff: header-only (INTERFACE) library")
endif()

# Keep the headers visible in IDE project trees regardless of library kind.
if(AUTODIFF_HEADERS)
  add_custom_target(autodiff_headers SOURCES ${AUTODIFF_HEADERS})
endif()

add_library(autodiff::autodiff ALIAS autodiff)
