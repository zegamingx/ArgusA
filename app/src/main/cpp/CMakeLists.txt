cmake_minimum_required(VERSION 3.28...4.5)

# Argus uses C++23 headers, not C++ standard-library modules. Keep the
# optional import-std path disabled for Android cross-compilation: CMake may
# otherwise discover host GCC module metadata instead of Android NDK libc++.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CXX_MODULE_STD OFF)

project(argus_training LANGUAGES CXX)

# Export a compile database for native diagnostics and static-analysis tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ARGUS_NDK_ROOT "${CMAKE_ANDROID_NDK}")
if(NOT ARGUS_NDK_ROOT AND DEFINED ANDROID_NDK)
    set(ARGUS_NDK_ROOT "${ANDROID_NDK}")
endif()
if(NOT ARGUS_NDK_ROOT)
    set(ARGUS_NDK_ROOT "$ENV{ANDROID_NDK_HOME}")
endif()
if(NOT ARGUS_NDK_ROOT)
    set(ARGUS_NDK_ROOT "$ENV{ANDROID_NDK_ROOT}")
endif()

if(NOT ARGUS_NDK_ROOT)
    message(FATAL_ERROR "Android NDK path is not available to the Vulkan shader build.")
endif()

# NDK shader-tools use host-specific subdirectories, e.g. linux-x86_64.
file(
    GLOB ARGUS_SHADER_TOOL_DIRS
    LIST_DIRECTORIES true
    "${ARGUS_NDK_ROOT}/shader-tools/*"
)

find_program(
    ARGUS_GLSLC
    NAMES glslc glslc.exe
    HINTS ${ARGUS_SHADER_TOOL_DIRS}
    NO_DEFAULT_PATH
)

if(NOT ARGUS_GLSLC)
    message(FATAL_ERROR
        "glslc was not found in ${ARGUS_NDK_ROOT}/shader-tools. "
        "Install the pinned Android NDK version."
    )
endif()

set(ARGUS_SPIRV_EMBED_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv_header.cmake"
)

function(argus_compile_shader source_file binary_name header_name array_symbol size_symbol)
    set(shader_target_environment "vulkan1.0")
    if(ARGC GREATER 5)
        set(shader_target_environment "${ARGV5}")
    endif()

    set(shader_binary "${CMAKE_CURRENT_BINARY_DIR}/${binary_name}")
    set(shader_header "${CMAKE_CURRENT_BINARY_DIR}/${header_name}")

    # Model shader compilation as a proper build rule. Unchanged shaders are
    # not recompiled on every CMake configure, and incremental builds only
    # regenerate the SPIR-V/header pair whose inputs actually changed.
    add_custom_command(
        OUTPUT
            "${shader_binary}"
            "${shader_header}"
        COMMAND
            "${ARGUS_GLSLC}"
            "--target-env=${shader_target_environment}"
            -O
            "${source_file}"
            -o "${shader_binary}"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DINPUT_FILE=${shader_binary}"
            "-DOUTPUT_FILE=${shader_header}"
            "-DARRAY_SYMBOL=${array_symbol}"
            "-DSIZE_SYMBOL=${size_symbol}"
            -P "${ARGUS_SPIRV_EMBED_SCRIPT}"
        DEPENDS
            "${source_file}"
            "${ARGUS_SPIRV_EMBED_SCRIPT}"
        COMMENT "Compiling Vulkan shader ${binary_name}"
        VERBATIM
    )

    set_property(
        GLOBAL
        APPEND
        PROPERTY ARGUS_SHADER_HEADERS "${shader_header}"
    )
endfunction()

argus_compile_shader(
    "${CMAKE_CURRENT_SOURCE_DIR}/adamw.comp"
    "adamw.comp.spv"
    "adamw_shader_spv.h"
    "kArgusAdamwShaderSpv"
    "kArgusAdamwShaderSpvSize"
)

argus_compile_shader(
    "${CMAKE_CURRENT_SOURCE_DIR}/linear.comp"
    "linear.comp.spv"
    "linear_shader_spv.h"
    "kArgusLinearShaderSpv"
    "kArgusLinearShaderSpvSize"
)

argus_compile_shader(
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_fp16.comp"
    "linear_fp16.comp.spv"
    "linear_fp16_shader_spv.h"
    "kArgusLinearFp16ShaderSpv"
    "kArgusLinearFp16ShaderSpvSize"
    "vulkan1.2"
)

argus_compile_shader(
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_resident.comp"
    "linear_resident.comp.spv"
    "linear_resident_shader_spv.h"
    "kArgusResidentLinearShaderSpv"
    "kArgusResidentLinearShaderSpvSize"
)

get_property(
    ARGUS_SHADER_HEADERS
    GLOBAL
    PROPERTY ARGUS_SHADER_HEADERS
)

add_custom_target(
    argus_shader_headers
    DEPENDS ${ARGUS_SHADER_HEADERS}
)

add_library(argus_training SHARED vulkan_training_jni.cpp)
add_dependencies(argus_training argus_shader_headers)

# Be explicit at target scope as well as directory scope. The backend does not
# use C++ named modules, and Android NDK r30's cross-toolchain must use libc++
# headers rather than host libstdc++ module interfaces.
set_target_properties(
    argus_training
    PROPERTIES
        CXX_SCAN_FOR_MODULES OFF
        CXX_MODULE_STD OFF
)

target_compile_features(argus_training PRIVATE cxx_std_23)
target_compile_options(argus_training PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(
    argus_training
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(
    argus_training
    PRIVATE
        vulkan
        android
        log
)
