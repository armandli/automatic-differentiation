# ---------------------------------------------------------------------------
#  Unit tests (GoogleTest, pulled in via FetchContent).
#
#  Every *.cpp directly under test/ is compiled into its own executable of the
#  same name, linked against the autodiff library and GoogleTest, and its test
#  cases are registered with CTest.
# ---------------------------------------------------------------------------

include(FetchContent)

set(AUTODIFF_GTEST_TAG "v1.18.0" CACHE STRING
  "GoogleTest git tag to fetch for the test suite")

# Prefer an already-installed GoogleTest only when explicitly asked; by default
# fetch and build a pinned copy so the test build is self-contained.
option(AUTODIFF_USE_SYSTEM_GTEST "Use a system-installed GoogleTest if present" OFF)
set(_gtest_find_args "")
if(AUTODIFF_USE_SYSTEM_GTEST)
  set(_gtest_find_args FIND_PACKAGE_ARGS NAMES GTest)
endif()

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        ${AUTODIFF_GTEST_TAG}
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
  SYSTEM
  ${_gtest_find_args}
)

# Don't let GoogleTest install itself or clutter our targets.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # harmless off-Windows

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

file(GLOB AUTODIFF_TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if(NOT AUTODIFF_TEST_SOURCES)
  message(WARNING
    "AUTODIFF_BUILD_TESTS is ON but no *.cpp files were found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

foreach(test_source IN LISTS AUTODIFF_TEST_SOURCES)
  get_filename_component(test_name "${test_source}" NAME_WE)

  add_executable(${test_name} "${test_source}")
  target_link_libraries(${test_name} PRIVATE
    autodiff::autodiff
    autodiff::warnings
    GTest::gtest_main
  )

  gtest_discover_tests(${test_name}
    PROPERTIES LABELS "unit"
  )
endforeach()
