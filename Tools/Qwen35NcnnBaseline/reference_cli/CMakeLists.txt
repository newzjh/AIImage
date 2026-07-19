cmake_minimum_required(VERSION 3.20)
project(qwen35_ncnn_reference_cli LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NCNN_SOURCE_DIR "" CACHE PATH "Path to a Tencent/ncnn master source checkout")
set(NCNN_LLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../ref/ncnn_llm-main" CACHE PATH "Path to ncnn_llm reference sources")
set(NLOHMANN_JSON_INCLUDE_DIR "" CACHE PATH "Directory containing nlohmann/json.hpp")

if(NOT EXISTS "${NCNN_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "NCNN_SOURCE_DIR must point to an ncnn source checkout")
endif()
if(NOT EXISTS "${NCNN_LLM_SOURCE_DIR}/src/ncnn_llm_gpt.cpp")
    message(FATAL_ERROR "NCNN_LLM_SOURCE_DIR does not contain the reference project")
endif()
if(NOT EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "NLOHMANN_JSON_INCLUDE_DIR must contain nlohmann/json.hpp")
endif()

set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_C_API OFF CACHE BOOL "" FORCE)
set(NCNN_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_OPENMP ON CACHE BOOL "" FORCE)
set(NCNN_PIXEL ON CACHE BOOL "" FORCE)
set(NCNN_RUNTIME_CPU OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
set(NCNN_AVX OFF CACHE BOOL "" FORCE)
set(NCNN_AVX2 OFF CACHE BOOL "" FORCE)
set(NCNN_AVX512 OFF CACHE BOOL "" FORCE)
set(NCNN_AVX512BF16 OFF CACHE BOOL "" FORCE)
set(NCNN_AVX512FP16 OFF CACHE BOOL "" FORCE)
set(NCNN_AVX512VNNI OFF CACHE BOOL "" FORCE)
set(NCNN_AVXNECONVERT OFF CACHE BOOL "" FORCE)
set(NCNN_AVXVNNI OFF CACHE BOOL "" FORCE)
set(NCNN_AVXVNNIINT8 OFF CACHE BOOL "" FORCE)
set(NCNN_AVXVNNIINT16 OFF CACHE BOOL "" FORCE)
set(NCNN_FMA OFF CACHE BOOL "" FORCE)
set(NCNN_XOP OFF CACHE BOOL "" FORCE)

# The six downloaded networks use only these built-in layer types. Keeping the
# registry exact avoids compiling every unrelated ncnn layer into the external
# gold executable; GatedDeltaRule and ShortConv are registered by ncnn_llm.
set(QWEN35_REQUIRED_NCNN_LAYERS
    BinaryOp
    Concat
    Convolution3D
    Embed
    ExpandDims
    Flatten
    GELU
    Gemm
    Input
    Interp
    LayerNorm
    MemoryData
    Packing
    Padding
    Permute
    RMSNorm
    Reshape
    RotaryEmbed
    SDPA
    Sigmoid
    Slice
    Softmax
    Split
    Swish
    Tile)
file(STRINGS "${NCNN_SOURCE_DIR}/src/CMakeLists.txt" NCNN_LAYER_DECLARATIONS
    REGEX "^[ \\t]*ncnn_add_layer\\(")
foreach(NCNN_LAYER_DECLARATION IN LISTS NCNN_LAYER_DECLARATIONS)
    string(REGEX REPLACE
        "^[ \\t]*ncnn_add_layer\\(([A-Za-z0-9_]+).*$"
        "\\1"
        NCNN_LAYER_NAME
        "${NCNN_LAYER_DECLARATION}")
    list(FIND QWEN35_REQUIRED_NCNN_LAYERS "${NCNN_LAYER_NAME}" NCNN_LAYER_REQUIRED_INDEX)
    string(TOLOWER "${NCNN_LAYER_NAME}" NCNN_LAYER_CACHE_KEY)
    if(NCNN_LAYER_REQUIRED_INDEX EQUAL -1)
        set("WITH_LAYER_${NCNN_LAYER_CACHE_KEY}" OFF CACHE BOOL "" FORCE)
    else()
        set("WITH_LAYER_${NCNN_LAYER_CACHE_KEY}" ON CACHE BOOL "" FORCE)
    endif()
endforeach()
add_subdirectory("${NCNN_SOURCE_DIR}" ncnn-build EXCLUDE_FROM_ALL)

file(GLOB TOKENIZER_SOURCES CONFIGURE_DEPENDS
    "${NCNN_LLM_SOURCE_DIR}/src/utils/tokenizer/*.cpp")
add_library(ncnn_tokenizer STATIC ${TOKENIZER_SOURCES})
target_include_directories(ncnn_tokenizer PUBLIC
    "${NCNN_LLM_SOURCE_DIR}/src"
    "${NLOHMANN_JSON_INCLUDE_DIR}")
target_link_libraries(ncnn_tokenizer PUBLIC ncnn)

file(GLOB NCNN_LLM_CORE_SOURCES CONFIGURE_DEPENDS
    "${NCNN_LLM_SOURCE_DIR}/src/*.cpp"
    "${NCNN_LLM_SOURCE_DIR}/src/utils/*.cpp")
add_library(ncnn_llm STATIC ${NCNN_LLM_CORE_SOURCES})
target_include_directories(ncnn_llm PUBLIC
    "${NCNN_LLM_SOURCE_DIR}/src"
    "${NLOHMANN_JSON_INCLUDE_DIR}")
target_link_libraries(ncnn_llm PUBLIC ncnn ncnn_tokenizer)

file(GLOB CLI_SOURCES CONFIGURE_DEPENDS
    "${NCNN_LLM_SOURCE_DIR}/examples/llm_ncnn_run/*.cpp")
add_executable(llm_ncnn_run ${CLI_SOURCES})
target_include_directories(llm_ncnn_run PRIVATE
    "${NCNN_LLM_SOURCE_DIR}/examples"
    "${NLOHMANN_JSON_INCLUDE_DIR}")
target_link_libraries(llm_ncnn_run PRIVATE ncnn_llm)

if(MSVC)
    target_compile_options(ncnn_tokenizer PRIVATE /utf-8)
    target_compile_options(ncnn_llm PRIVATE /utf-8)
    target_compile_options(llm_ncnn_run PRIVATE /utf-8)
    target_compile_definitions(ncnn_tokenizer PRIVATE NOMINMAX)
    target_compile_definitions(ncnn_llm PRIVATE NOMINMAX)
    target_compile_definitions(llm_ncnn_run PRIVATE NOMINMAX)
endif()
