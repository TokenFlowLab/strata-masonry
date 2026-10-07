# books/code Makefile -- builds and runs primitives/composites/blocks tests.
#
# Usage:
#   make <N>_test             -- build and run one test (e.g. make 0_tcgen05_alloc_test)
#   make tests_sm100a         -- all sm_100a tests (runs on GB200, also on GB300)
#   make tests_sm103a         -- all sm_103a tests (runs on GB300 only)
#   make tests_sm90a          -- all sm_90a tests (runs on H100, also compiles on Blackwell)
#   make all                  -- build (but do not run) every test binary
#   make clean                -- remove build/ and .nvcc_tmp/
#
# blocks/ is a header-only subtree; each block's parametric kernel lives in
# blocks/<N>_<name>.cuh and is exercised via tests/<N>_<name>_test.cu. There
# are no standalone block binaries.
#
# Test arch is determined by a "// ARCH: sm_XXXa" comment within the first
# 10 lines of the .cu file. Default is sm_100a when no ARCH comment is
# present.

SHELL := /bin/bash
NVCC  ?= nvcc

CUTLASS_DIR := ../../dynamic-kernel-generator/cutlass
CXXFLAGS := -O3 -std=c++17 --expt-relaxed-constexpr --extended-lambda -DNDEBUG -lineinfo \
            -I primitives -I composites -I blocks -I tests \
            -I $(CUTLASS_DIR)/include -I $(CUTLASS_DIR)/tools/util/include
LDFLAGS  := -lcuda -lcublas
EXTRA_CXXFLAGS ?=
EXTRA_LDFLAGS  ?=
CXXFLAGS += $(EXTRA_CXXFLAGS)
LDFLAGS  += $(EXTRA_LDFLAGS)

CODEDIR  := $(abspath .)
BUILDDIR := $(CODEDIR)/build
TMPDIR   := $(CODEDIR)/.nvcc_tmp
CXXFLAGS += -DPL_REF_CACHE_DIR=\"$(CODEDIR)/verify_cache\"

# Enumerate tests -- anything matching tests/*_test.cu
TESTS      := $(wildcard tests/*_test.cu)
TEST_NAMES := $(patsubst tests/%.cu,%,$(TESTS))

# A file whose top-level guard is `#if defined(PL_AGENTIC_SM90A)` (no SM100A
# fallback) is Hopper-only. On non-SM90A builds it has no main(), so we
# filter it out before declaring build targets.
is_hopper_only_file = $(shell head -1 $(1) 2>/dev/null | grep -qE '^\#if defined\(PL_AGENTIC_SM90A\)$$' && echo 1)
HOPPER_ONLY_TEST_NAMES := $(strip $(foreach t,$(TEST_NAMES),$(if $(call is_hopper_only_file,tests/$(t).cu),$(t))))

# A file is Blackwell-only if its TOP-LINE guard is `#if defined(PL_AGENTIC_SM100A)`
# (and the file has no SM90A path). After unification, fully merged files have
# no guard at line 1 and should NOT be filtered.
is_blackwell_only_file = $(shell head -1 $(1) 2>/dev/null | grep -qE '^\#if defined\(PL_AGENTIC_SM100A\)' && ! grep -qE '^\#if defined\(PL_AGENTIC_SM90A\)' $(1) 2>/dev/null && echo 1)
BLACKWELL_ONLY_TEST_NAMES := $(strip $(foreach t,$(TEST_NAMES),$(if $(call is_blackwell_only_file,tests/$(t).cu),$(t))))

# A file is SM103A-only if its TOP-LINE guard is `#if defined(PL_AGENTIC_SM103A)`
# and it has no SM100A or SM90A path. These files use sm_103a-exclusive PTX
# (e.g. K=96 MMA atom, byte-address SMEM descriptor) and do not
# compile on sm_100a or sm_90a, so they must be filtered out of those builds.
is_sm103a_only_file = $(shell head -1 $(1) 2>/dev/null | grep -qE '^\#if defined\(PL_AGENTIC_SM103A\)$$' && ! grep -qE '^\#if defined\(PL_AGENTIC_SM(100A|90A)\)' $(1) 2>/dev/null && echo 1)
SM103A_ONLY_TEST_NAMES := $(strip $(foreach t,$(TEST_NAMES),$(if $(call is_sm103a_only_file,tests/$(t).cu),$(t))))

# A file is SM100A-strict (no SM103A path) if its TOP-LINE guard is EXACTLY
# `#if defined(PL_AGENTIC_SM100A)` (no `|| defined(PL_AGENTIC_SM103A)` after).
# These files use instructions removed on Blackwell Ultra (e.g. kind::i8 MMA)
# and won't compile when targeting sm_103a -- drop them on GB300 hosts.
is_sm100a_only_file = $(shell head -1 $(1) 2>/dev/null | grep -qE '^\#if defined\(PL_AGENTIC_SM100A\)$$' && echo 1)
SM100A_ONLY_TEST_NAMES := $(strip $(foreach t,$(TEST_NAMES),$(if $(call is_sm100a_only_file,tests/$(t).cu),$(t))))

# Bucket tests by ARCH comment. Scan first 10 lines so the ARCH: marker can
# appear after a leading `#if defined(PL_AGENTIC_SM100A)` guard.
define arch_of
$(shell awk 'NR<=10 && /ARCH:/ { for (i=1;i<=NF;i++) if ($$i ~ /^sm_/) { print $$i; exit } } NR>10 { exit }' tests/$(1).cu 2>/dev/null)
endef

SM100A_TESTS := $(strip $(foreach t,$(TEST_NAMES),$(if $(filter sm_100a,$(call arch_of,$(t))),$(t))))
SM103A_TESTS := $(strip $(foreach t,$(TEST_NAMES),$(if $(filter sm_103a,$(call arch_of,$(t))),$(t))))
SM90A_TESTS  := $(strip $(foreach t,$(TEST_NAMES),$(if $(filter sm_90a,$(call arch_of,$(t))),$(t))))
# Tests that don't declare ARCH default to sm_100a.
NOARCH_TESTS := $(strip $(foreach t,$(TEST_NAMES),$(if $(call arch_of,$(t)),,$(t))))
SM100A_TESTS += $(NOARCH_TESTS)

.PHONY: all clean build_all tests_sm100a tests_sm103a tests_sm90a $(TEST_NAMES)

# Filter active build set per NATIVE_ARCH. A file's main() lives behind its
# top-level guard; if PL_AGENTIC_<arch> is not defined the TU has no main and
# the link step would fail. Drop arch-incompatible files before declaring
# build targets.
ifeq ($(NATIVE_ARCH),sm_90a)
  # Hopper: drop Blackwell-only and SM103A-only files.
  ACTIVE_TEST_NAMES := $(filter-out $(BLACKWELL_ONLY_TEST_NAMES) $(SM103A_ONLY_TEST_NAMES),$(TEST_NAMES))
else ifeq ($(NATIVE_ARCH),sm_103a)
  # GB300: keep Blackwell-base files (SM100A || SM103A) and SM103A-only files;
  # drop Hopper-only and SM100A-strict files (the latter use instructions
  # removed on sm_103a like tcgen05.mma.kind::i8).
  ACTIVE_TEST_NAMES := $(filter-out $(HOPPER_ONLY_TEST_NAMES) $(SM100A_ONLY_TEST_NAMES),$(TEST_NAMES))
else
  # GB200 / sm_100a default: drop Hopper-only and SM103A-only files (the
  # latter use instructions absent from sm_100a like K=96 MMA atom,
  # byte-address SMEM desc).
  ACTIVE_TEST_NAMES := $(filter-out $(HOPPER_ONLY_TEST_NAMES) $(SM103A_ONLY_TEST_NAMES),$(TEST_NAMES))
endif

all: $(addprefix $(BUILDDIR)/,$(ACTIVE_TEST_NAMES))
build_all: all

# `make <name>_test` builds and runs it.
$(TEST_NAMES): %: $(BUILDDIR)/%
	@echo "=== running $@ ==="
	@$(BUILDDIR)/$@

# `make <name>_build` builds without running (sandbox-friendly).
BUILD_NAMES := $(addsuffix _build,$(TEST_NAMES))
.PHONY: $(BUILD_NAMES)
$(BUILD_NAMES): %_build: $(BUILDDIR)/%
	@echo "[build OK] $<"

# Native GPU arch: override via `make NATIVE_ARCH=sm_103a ...` on GB300.
# On GB200 the default sm_100a is correct.
NATIVE_ARCH ?= sm_100a

# api_def reflects the **target platform** (NATIVE_ARCH). Source files use
# guards like `#if defined(PL_AGENTIC_SM100A) || ...` so that, e.g., a
# Blackwell-only primitive becomes an empty TU when built on Hopper.
ifeq ($(NATIVE_ARCH),sm_90a)
  API_DEF := -DPL_AGENTIC_SM90A
else ifeq ($(NATIVE_ARCH),sm_103a)
  API_DEF := -DPL_AGENTIC_SM103A
else
  API_DEF := -DPL_AGENTIC_SM100A
endif

# Per-test build rule. Compiles for the ARCH comment's arch (SASS + PTX),
# and ALSO for NATIVE_ARCH so the binary runs on the current host GPU.
# The ARCH: annotation is "minimum required feature level" -- for cross-arch
# targets (e.g. sm_90a test on Blackwell host) we emit a second compilation
# at NATIVE_ARCH to get native SASS.
# Hopper-only tests: contain WGMMA features that don't exist on Blackwell.
# Compile these for sm_90a only (they won't run on Blackwell).
HOPPER_ONLY_PATTERN := wgmma
is_hopper_only = $(shell grep -lE 'wgmma\.mma_async|wgmma\.fence|wgmma\.commit_group|wgmma\.wait_group|wgmma_f16|wgmma_bf16|wgmma_fp8|wgmma_tf32|wgmma_s8|wgmma_fence_commit_wait|mma_warp_hopper' $(1) 2>/dev/null)

$(BUILDDIR)/%_test: tests/%_test.cu | $(BUILDDIR) $(TMPDIR)
	@arch=$$(awk 'NR<=10 && /ARCH:/ { for (i=1;i<=NF;i++) if ($$i ~ /^sm_/) { print $$i; exit } } NR>10 { exit }' $<); \
	  if [ -z "$$arch" ]; then arch=$(NATIVE_ARCH); fi; \
	  cc=$${arch#sm_}; \
	  native=$(NATIVE_ARCH); native=$${native#sm_}; \
	  hopper_only=$$(grep -lE 'wgmma_f16|wgmma_bf16|wgmma_fp8|wgmma_tf32|wgmma_s8|wgmma_fence|k_loop_hopper|mma_warp_hopper|59_wgmma_fence|53_wgmma|54_wgmma|55_wgmma|56_wgmma|57_wgmma|58_wgmma' $< 2>/dev/null); \
	  if [ "$$cc" = "$$native" ]; then \
	    echo "[nvcc sm_$${cc} $(API_DEF)] $@"; \
	    TMPDIR=$(TMPDIR) $(NVCC) \
	      -gencode arch=compute_$${cc},code=sm_$${cc} \
	      -gencode arch=compute_$${cc},code=compute_$${cc} \
	      $(API_DEF) $(CXXFLAGS) -MMD -MT $@ -MF $@.d -o $@ $< $(LDFLAGS); \
	  elif [ -n "$$hopper_only" ]; then \
	    echo "[nvcc sm_$${cc} only -- wgmma not on native] $@"; \
	    TMPDIR=$(TMPDIR) $(NVCC) \
	      -gencode arch=compute_$${cc},code=sm_$${cc} \
	      -gencode arch=compute_$${cc},code=compute_$${cc} \
	      $(API_DEF) $(CXXFLAGS) -MMD -MT $@ -MF $@.d -o $@ $< $(LDFLAGS); \
	  else \
	    echo "[nvcc sm_$${cc}+sm_$${native} $(API_DEF)] $@"; \
	    TMPDIR=$(TMPDIR) $(NVCC) \
	      -gencode arch=compute_$${cc},code=sm_$${cc} \
	      -gencode arch=compute_$${native},code=sm_$${native} \
	      -gencode arch=compute_$${native},code=compute_$${native} \
	      $(API_DEF) $(CXXFLAGS) -MMD -MT $@ -MF $@.d -o $@ $< $(LDFLAGS); \
	  fi

$(BUILDDIR) $(TMPDIR):
	@mkdir -p $@

tests_sm100a: $(addprefix $(BUILDDIR)/,$(SM100A_TESTS))
	@failed=0; for t in $^; do echo "=== running $$t ==="; $$t || failed=$$((failed+1)); done; \
	  echo "sm_100a: $$((${words $(SM100A_TESTS)}-failed))/$(words $(SM100A_TESTS)) passed"; \
	  [ $$failed -eq 0 ]

tests_sm103a: $(addprefix $(BUILDDIR)/,$(SM103A_TESTS))
	@failed=0; for t in $^; do echo "=== running $$t ==="; $$t || failed=$$((failed+1)); done; \
	  echo "sm_103a: $$((${words $(SM103A_TESTS)}-failed))/$(words $(SM103A_TESTS)) passed"; \
	  [ $$failed -eq 0 ]

tests_sm90a: $(addprefix $(BUILDDIR)/,$(SM90A_TESTS))
	@failed=0; for t in $^; do echo "=== running $$t ==="; $$t || failed=$$((failed+1)); done; \
	  echo "sm_90a: $$((${words $(SM90A_TESTS)}-failed))/$(words $(SM90A_TESTS)) passed"; \
	  [ $$failed -eq 0 ]

# ============================================================================
# kernels/ -- V0 deliverable kernels per Phase 5(c).
#
# Layout: code/kernels/sm{100a,103a,90a}/<name>.cu, one end-to-end .cu per
# kernel (kernel + driver + verify + bench + main). Output binary
# build/kernel_<arch>_<name>.
#
# Per-arch glob; matches NATIVE_ARCH so an agent only builds their arch:
#   make kernels                 -> all kernels for NATIVE_ARCH
#   make kernel_<arch>_<name>    -> build + run one kernel
#   make kernel_<arch>_<name>_build -> build only (sandbox-friendly)
# ============================================================================

# Default-empty so missing dirs don't break the wildcard. Kernels are
# organized under kernels/<family>/<arch>/ where <family> is gemm,
# fmha, grouped_gemm, etc.
KERNELS_SM100A := $(wildcard kernels/*/sm100a/*.cu) $(wildcard kernels/*/sm100a/nvfp4/*.cu) $(wildcard kernels/*/sm100a/backward/*.cu)
KERNELS_SM103A := $(wildcard kernels/*/sm103a/*.cu)
KERNELS_SM90A  := $(wildcard kernels/*/sm90a/*.cu)

# Active set per NATIVE_ARCH.
ifeq ($(NATIVE_ARCH),sm_90a)
  ACTIVE_KERNELS := $(KERNELS_SM90A)
  KERNEL_ARCH_TAG := sm90a
else ifeq ($(NATIVE_ARCH),sm_103a)
  ACTIVE_KERNELS := $(KERNELS_SM103A)
  KERNEL_ARCH_TAG := sm103a
else
  ACTIVE_KERNELS := $(KERNELS_SM100A)
  KERNEL_ARCH_TAG := sm100a
endif

# Strip the family prefix to produce kernel_<arch>_<name> targets. The
# family stays implicit in the source path (resolved via vpath below)
# so target names don't collide if two families ship a kernel with the
# same .cu basename -- if that happens, switch to kernel_<family>_<arch>_<name>.
KERNEL_NAMES   := $(patsubst %.cu,kernel_$(KERNEL_ARCH_TAG)_%,$(notdir $(ACTIVE_KERNELS)))
KERNEL_BUILDS  := $(addsuffix _build,$(KERNEL_NAMES))

.PHONY: kernels $(KERNEL_NAMES) $(KERNEL_BUILDS)

# Build-and-run all active kernels for NATIVE_ARCH.
kernels: $(addprefix $(BUILDDIR)/,$(KERNEL_NAMES))
	@for k in $^; do echo "=== running $$k ==="; $$k || exit 1; done
	@echo "kernels ($(KERNEL_ARCH_TAG)): $(words $(KERNEL_NAMES)) ran"

# `make kernel_<arch>_<name>` -> build + run one kernel.
$(KERNEL_NAMES): %: $(BUILDDIR)/%
	@echo "=== running $@ ==="
	@$(BUILDDIR)/$@

# `make kernel_<arch>_<name>_build` -> build only.
$(KERNEL_BUILDS): %_build: $(BUILDDIR)/%
	@echo "[build OK] $<"

# Resolve kernel_<arch>_<name> back to its source under kernels/*/<arch>/.
vpath %.cu $(sort $(dir $(ACTIVE_KERNELS)))

# Per-kernel build rule. Compiles with NATIVE_ARCH only (kernels are
# arch-specific deliverables; no cross-arch SASS fallback like tests/).
$(BUILDDIR)/kernel_$(KERNEL_ARCH_TAG)_%: %.cu | $(BUILDDIR) $(TMPDIR)
	@cc=$(NATIVE_ARCH); cc=$${cc#sm_}; \
	  echo "[nvcc sm_$${cc} $(API_DEF)] $@"; \
	  TMPDIR=$(TMPDIR) $(NVCC) \
	    -gencode arch=compute_$${cc},code=sm_$${cc} \
	    -gencode arch=compute_$${cc},code=compute_$${cc} \
	    $(API_DEF) $(CXXFLAGS) -MMD -MT $@ -MF $@.d -o $@ $< $(LDFLAGS)

# Header-dependency tracking: nvcc -MMD writes a $@.d listing every included
# .cuh/.cu. Pulling them in makes an edit to any header rebuild its dependents
# (without this, the build rule only sees the top-level .cu and silently skips
# recompiles when only a block/composite header changed).
-include $(wildcard $(BUILDDIR)/*.d)

clean:
	rm -rf $(BUILDDIR) $(TMPDIR)
