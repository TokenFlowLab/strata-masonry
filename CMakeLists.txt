cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES OFF)
project(gpu_kernels LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(GPU_ARCH "sm_100a" CACHE STRING "Target GPU architecture: sm_100a, sm_103a or sm_90a")
set_property(CACHE GPU_ARCH PROPERTY STRINGS sm_100a sm_103a sm_90a)
option(BUILD_TESTS "Build primitive/composite/block tests" ON)
option(BUILD_KERNELS "Build kernel drivers" ON)

if(GPU_ARCH STREQUAL "sm_100a")
  set(API_DEF PL_AGENTIC_SM100A)
  set(ARCH_TAG sm100a)
elseif(GPU_ARCH STREQUAL "sm_103a")
  set(API_DEF PL_AGENTIC_SM103A)
  set(ARCH_TAG sm103a)
elseif(GPU_ARCH STREQUAL "sm_90a")
  set(API_DEF PL_AGENTIC_SM90A)
  set(ARCH_TAG sm90a)
else()
  message(FATAL_ERROR "GPU_ARCH must be sm_100a, sm_103a or sm_90a (got ${GPU_ARCH})")
endif()
string(REPLACE "sm_" "" NATIVE_CC "${GPU_ARCH}")

set(REF_CACHE_DIR "${CMAKE_BINARY_DIR}/verify_cache")

function(add_cuda_binary target source)
  set(ccs ${ARGN})
  add_executable(${target} ${source})
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/primitives ${PROJECT_SOURCE_DIR}/src/composites
    ${PROJECT_SOURCE_DIR}/src/blocks ${PROJECT_SOURCE_DIR}/tests)
  target_compile_definitions(${target} PRIVATE
    ${API_DEF} NDEBUG PL_REF_CACHE_DIR="${REF_CACHE_DIR}")
  list(GET ccs -1 ptx_cc)
  set(gencode "")
  foreach(cc IN LISTS ccs)
    list(APPEND gencode "SHELL:-gencode arch=compute_${cc},code=sm_${cc}")
  endforeach()
  list(APPEND gencode "SHELL:-gencode arch=compute_${ptx_cc},code=compute_${ptx_cc}")
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -std=c++17 --expt-relaxed-constexpr --extended-lambda -lineinfo ${gencode}>)
  target_link_libraries(${target} PRIVATE CUDA::cuda_driver)
endfunction()

# Tests: arch from a "// ARCH: sm_XXXa" comment in the first 10 lines (default GPU_ARCH).
# Line-1 guards mark files that only build for one arch; drop those that cannot build here.
if(BUILD_TESTS)
  enable_testing()
  file(GLOB test_sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/*_test.cu)
  foreach(source IN LISTS test_sources)
    get_filename_component(name ${source} NAME_WE)
    file(STRINGS ${source} head LIMIT_COUNT 10)
    list(GET head 0 line1)
    file(READ ${source} body)

    set(skip OFF)
    if(line1 MATCHES "^#if defined\\(PL_AGENTIC_SM90A\\)$" AND NOT GPU_ARCH STREQUAL "sm_90a")
      set(skip ON)
    endif()
    if(line1 MATCHES "^#if defined\\(PL_AGENTIC_SM103A\\)$" AND NOT body MATCHES "#if defined\\(PL_AGENTIC_SM(100A|90A)\\)"
       AND NOT GPU_ARCH STREQUAL "sm_103a")
      set(skip ON)
    endif()
    if(line1 MATCHES "^#if defined\\(PL_AGENTIC_SM100A\\)$" AND GPU_ARCH STREQUAL "sm_103a")
      set(skip ON)
    endif()
    if(line1 MATCHES "^#if defined\\(PL_AGENTIC_SM100A\\)" AND NOT body MATCHES "#if defined\\(PL_AGENTIC_SM90A\\)"
       AND GPU_ARCH STREQUAL "sm_90a")
      set(skip ON)
    endif()
    if(skip)
      continue()
    endif()

    set(test_cc ${NATIVE_CC})
    foreach(line IN LISTS head)
      string(FIND "${line}" "ARCH:" pos)
      if(pos GREATER -1)
        string(SUBSTRING "${line}" ${pos} -1 tail)
        if(tail MATCHES "sm_([0-9]+a)")
          set(test_cc ${CMAKE_MATCH_1})
          break()
        endif()
      endif()
    endforeach()

    if(test_cc STREQUAL NATIVE_CC)
      add_cuda_binary(${name} ${source} ${test_cc})
    elseif(body MATCHES "wgmma_f16|wgmma_bf16|wgmma_fp8|wgmma_tf32|wgmma_s8|wgmma_fence|k_loop_hopper|mma_warp_hopper|59_wgmma_fence|5[3-8]_wgmma")
      add_cuda_binary(${name} ${source} ${test_cc})
    else()
      add_cuda_binary(${name} ${source} ${test_cc} ${NATIVE_CC})
    endif()
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
endif()

# Kernels: src/kernels/<family>/<arch>/{,nvfp4/,backward/}*.cu -> kernel_<arch>_<name>, native arch only.
if(BUILD_KERNELS)
  file(GLOB kernel_sources CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/kernels/*/${ARCH_TAG}/*.cu
    ${PROJECT_SOURCE_DIR}/src/kernels/*/${ARCH_TAG}/nvfp4/*.cu
    ${PROJECT_SOURCE_DIR}/src/kernels/*/${ARCH_TAG}/backward/*.cu)
  foreach(source IN LISTS kernel_sources)
    get_filename_component(name ${source} NAME_WE)
    add_cuda_binary(kernel_${ARCH_TAG}_${name} ${source} ${NATIVE_CC})
  endforeach()
endif()
