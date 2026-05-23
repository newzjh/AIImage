cmake_minimum_required(VERSION 3.20)
project(realesrgan_unity CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_REALESRGAN_UNITY_DEFAULT_SHARED ON)
if (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_REALESRGAN_UNITY_DEFAULT_SHARED OFF)
endif()
option(REALESRGAN_UNITY_SHARED "Build realesrgan_unity as shared library" ${_REALESRGAN_UNITY_DEFAULT_SHARED})

if (REALESRGAN_UNITY_SHARED)
    add_library(realesrgan_unity SHARED realesrgan_unity.cpp)
else()
    add_library(realesrgan_unity STATIC realesrgan_unity.cpp)
endif()

target_compile_definitions(realesrgan_unity PRIVATE REALESRGAN_UNITY_BUILD=1)

if (MSVC)
    target_compile_options(realesrgan_unity PRIVATE /EHsc)
endif()

set(NCNN_ROOT "" CACHE PATH "Path to prebuilt ncnn root for the current platform")

if (NCNN_ROOT STREQUAL "")
    if (WIN32)
        set(NCNN_ROOT "${CMAKE_CURRENT_LIST_DIR}/ncnn-20260113-windows-vs2022-shared/x64")
    elseif (ANDROID)
        set(NCNN_ROOT "${CMAKE_CURRENT_LIST_DIR}/ncnn-20260113-android-vulkan-shared/${ANDROID_ABI}")
    elseif (APPLE)
        # iOS uses frameworks; macOS uses frameworks
        if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
            set(NCNN_ROOT "${CMAKE_CURRENT_LIST_DIR}/ncnn-20260113-ios-vulkan")
        else()
            set(NCNN_ROOT "${CMAKE_CURRENT_LIST_DIR}/ncnn-20260113-macos-vulkan")
        endif()
    endif()
endif()

if (WIN32)
    if (NOT EXISTS "${NCNN_ROOT}/include/ncnn/net.h")
        message(FATAL_ERROR "NCNN_ROOT invalid (missing include): ${NCNN_ROOT}")
    endif()

    add_library(ncnn SHARED IMPORTED GLOBAL)
    set_target_properties(ncnn PROPERTIES
        IMPORTED_IMPLIB "${NCNN_ROOT}/lib/ncnn.lib"
        IMPORTED_LOCATION "${NCNN_ROOT}/bin/ncnn.dll"
    )
    target_include_directories(realesrgan_unity PRIVATE "${NCNN_ROOT}/include")
    target_link_libraries(realesrgan_unity PRIVATE ncnn)
elseif (ANDROID)
    if (NOT EXISTS "${NCNN_ROOT}/include/ncnn/net.h")
        message(FATAL_ERROR "NCNN_ROOT invalid (missing include): ${NCNN_ROOT}")
    endif()
    add_library(ncnn SHARED IMPORTED GLOBAL)
    set_target_properties(ncnn PROPERTIES
        IMPORTED_LOCATION "${NCNN_ROOT}/lib/libncnn.so"
    )
    target_include_directories(realesrgan_unity PRIVATE "${NCNN_ROOT}/include")
    target_link_libraries(realesrgan_unity PRIVATE ncnn)
elseif (APPLE)
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(_NCNN_FRAMEWORK "${NCNN_ROOT}/ncnn.framework")
        set(_GLSLANG_FRAMEWORK "${NCNN_ROOT}/glslang.framework")
        set(_OPENMP_FRAMEWORK "${NCNN_ROOT}/openmp.framework")
    else()
        set(_NCNN_FRAMEWORK "${NCNN_ROOT}/ncnn.framework")
        set(_GLSLANG_FRAMEWORK "${NCNN_ROOT}/glslang.framework")
        set(_OPENMP_FRAMEWORK "${NCNN_ROOT}/openmp.framework")
    endif()

    if (NOT EXISTS "${_NCNN_FRAMEWORK}")
        message(FATAL_ERROR "ncnn.framework not found under: ${NCNN_ROOT}")
    endif()

    target_include_directories(realesrgan_unity PRIVATE "${_NCNN_FRAMEWORK}/Headers")
    target_link_libraries(realesrgan_unity PRIVATE
        "${_NCNN_FRAMEWORK}"
        "${_GLSLANG_FRAMEWORK}"
        "${_OPENMP_FRAMEWORK}"
    )
else()
    message(FATAL_ERROR "Unsupported platform for this build (WIN32/ANDROID/APPLE only)")
endif()

set_target_properties(realesrgan_unity PROPERTIES OUTPUT_NAME "realesrgan_unity")
