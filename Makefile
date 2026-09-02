# ===========================================================================
#  Convenience wrapper around CMake.
#
#  The build itself is described by the CMakeLists.txt files. This Makefile just
#  memorizes how to drive CMake so day to day you only type:
#
#     make            # configure (first time) + build everything
#     make test       # build + run the test suite via CTest
#     make clean       # remove build products, keep the configured tree
#     make distclean   # delete the build directory entirely
#     make rebuild     # distclean + build from scratch
#     make help        # list all targets
#
#  Overridable knobs:  make BUILD_TYPE=Release   make GENERATOR="Unix Makefiles"
#                      make MLX=ON   (AUTO by default: on when MLX is installed)
# ===========================================================================

BUILD_DIR  ?= build
BUILD_TYPE ?= Debug
GENERATOR  ?= $(shell command -v ninja >/dev/null 2>&1 && echo Ninja || echo "Unix Makefiles")
JOBS       ?= $(shell (command -v nproc >/dev/null 2>&1 && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)
MLX        ?= AUTO

CMAKE ?= cmake
CTEST ?= ctest

CACHE := $(BUILD_DIR)/CMakeCache.txt

CONFIGURE_FLAGS := -S . -B $(BUILD_DIR) -G "$(GENERATOR)" \
                   -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
                   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                   -DAUTODIFF_WITH_MLX=$(MLX)

.PHONY: all build configure reconfigure test clean distclean rebuild compdb help
.DEFAULT_GOAL := all

all: build

## all         : (default) configure if needed, then build
## build       : compile every target
build: $(CACHE)
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

## configure   : run CMake to generate the build system (only if not done yet)
configure: $(CACHE)

$(CACHE): CMakeLists.txt src/CMakeLists.txt test/CMakeLists.txt
	$(CMAKE) $(CONFIGURE_FLAGS)
	@$(MAKE) --no-print-directory compdb

## reconfigure : re-run CMake even if the build is already configured
reconfigure:
	$(CMAKE) $(CONFIGURE_FLAGS)
	@$(MAKE) --no-print-directory compdb

## test        : build, then run all tests through CTest
test: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure -j $(JOBS)

## clean       : remove build products but keep the CMake configuration
clean:
	@if [ -f "$(CACHE)" ]; then \
		$(CMAKE) --build $(BUILD_DIR) --target clean; \
	else \
		echo "Nothing to clean ($(BUILD_DIR)/ is not configured)"; \
	fi

## distclean   : delete the entire build directory
distclean:
	$(RM) -r $(BUILD_DIR) compile_commands.json

## rebuild     : distclean followed by a fresh configure + build
rebuild: distclean build

## compdb      : link compile_commands.json into the repo root (for clangd)
compdb:
	@if [ -f "$(BUILD_DIR)/compile_commands.json" ]; then \
		ln -sf $(BUILD_DIR)/compile_commands.json compile_commands.json; \
	fi

## help        : show this list of targets
help:
	@echo "Targets:"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed -e 's/^## /  /'
	@echo ""
	@echo "Current settings (override on the command line):"
	@echo "  BUILD_DIR  = $(BUILD_DIR)"
	@echo "  BUILD_TYPE = $(BUILD_TYPE)"
	@echo "  GENERATOR  = $(GENERATOR)"
	@echo "  JOBS       = $(JOBS)"
	@echo "  MLX        = $(MLX)"
