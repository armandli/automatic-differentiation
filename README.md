# automatic-differentiation

Learning how to implement an automatic differentiation framework in C++, as a
hobby project.

## Status

Early days. Currently there is a header-only forward-mode core built on dual
numbers (`ad::dual<T>`), with the usual arithmetic operators and a handful of
elementary functions (`sin`, `cos`, `exp`, `log`, `sqrt`, `pow`, ...).

```cpp
#include <forward_ad.h>
#include <print>

int main() {
  // f(x) = x^2 + 3x + 2   ->   f'(2) == 7
  auto f = [](auto x) { return x * x + 3.0 * x + 2.0; };
  std::println("f'(2) = {}", ad::derivative(f, 2.0));
}
```

## Layout

| Path    | Contents                                                        |
| ------- | -------------------------------------------------------------- |
| `src/`  | the library. All source lives here; headers are under `src/autodiff/` and included bare, e.g. `<dual_number.h>` (that directory is the include root). Header-only for now, but if any `.cpp` shows up here the build turns it into a static `libautodiff.a` automatically. |
| `test/` | GoogleTest unit tests. Every `*.cpp` here becomes its own test executable. |
| `build/`| out-of-source build tree (git-ignored). All built executables land directly in here. |

## Building

Requires **CMake ≥ 3.25**, a **C++26** compiler, and (optionally) **Ninja**.
GoogleTest is downloaded automatically at configure time via CMake's
`FetchContent`.

An optional **MLX** backend (Apple's array framework) is picked up automatically
when MLX is installed — `brew install mlx`, or `pip install mlx` and then
configure with `-DCMAKE_PREFIX_PATH=$(python3 -m mlx --cmake-dir)`. It is not
built from source; see `AUTODIFF_WITH_MLX` below.

Everything goes through the `Makefile`, which just memorizes the CMake commands:

```sh
make            # configure (first run) + build everything
make test       # build, then run the test suite through CTest
make clean      # remove build products, keep the CMake configuration
make distclean  # delete the build/ directory entirely
make rebuild    # distclean + fresh build
make help       # list targets and current settings
```

Knobs (override on the command line):

```sh
make BUILD_TYPE=Release          # default: Debug
make GENERATOR="Unix Makefiles"  # default: Ninja if available
make MLX=ON                      # default: AUTO (also: OFF)
make -j
```

Running the raw CMake commands instead:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

### Useful CMake options

| Option                          | Default | Effect                                        |
| ------------------------------- | ------- | --------------------------------------------- |
| `AUTODIFF_BUILD_TESTS`          | `ON`    | build the `test/` suite                       |
| `AUTODIFF_WARNINGS_AS_ERRORS`   | `OFF`   | add `-Werror` to first-party targets          |
| `AUTODIFF_WITH_MLX`             | `AUTO`  | MLX backend: `AUTO` (on if found), `ON` (required), `OFF` |
| `AUTODIFF_USE_SYSTEM_GTEST`     | `OFF`   | use an installed GoogleTest instead of fetching |
| `AUTODIFF_GTEST_TAG`            | `v1.18.0` | GoogleTest git tag to fetch                 |

A `compile_commands.json` is generated in `build/` and symlinked into the repo
root for `clangd`.
