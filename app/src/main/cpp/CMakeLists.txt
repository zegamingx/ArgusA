cmake_minimum_required(VERSION 3.22.1)

project(argus_training LANGUAGES CXX)

set(ARGUS_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/adamw.comp")
set(ARGUS_SHADER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/adamw.comp.spv")
set(ARGUS_SHADER_HEADER "${CMAKE_CURRENT_BINARY_DIR}/adamw_shader_spv.h")

set_property(
    DIRECTORY APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARGUS_SHADER_SOURCE}"
)

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

# The NDK stores shader-tools in host-specific subdirectories such as
# shader-tools/linux-x86_64 and shader-tools/windows-x86_64.
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

execute_process(
    COMMAND
        "${ARGUS_GLSLC}"
        --target-env=vulkan1.0
        -O
        "${ARGUS_SHADER_SOURCE}"
        -o "${ARGUS_SHADER_BINARY}"
    RESULT_VARIABLE ARGUS_GLSLC_RESULT
    OUTPUT_VARIABLE ARGUS_GLSLC_OUTPUT
    ERROR_VARIABLE ARGUS_GLSLC_ERROR
)

if(NOT ARGUS_GLSLC_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Vulkan shader compilation failed.\n"
        "stdout: ${ARGUS_GLSLC_OUTPUT}\n"
        "stderr: ${ARGUS_GLSLC_ERROR}"
    )
endif()

file(READ "${ARGUS_SHADER_BINARY}" ARGUS_SHADER_HEX HEX)
string(LENGTH "${ARGUS_SHADER_HEX}" ARGUS_SHADER_HEX_LENGTH)
math(EXPR ARGUS_SHADER_BYTE_COUNT "${ARGUS_SHADER_HEX_LENGTH} / 2")

if(ARGUS_SHADER_BYTE_COUNT LESS 20)
    message(FATAL_ERROR "The compiled Vulkan shader is incomplete.")
endif()

file(
    WRITE "${ARGUS_SHADER_HEADER}"
    "#pragma once\n"
    "#include <cstddef>\n"
    "#include <cstdint>\n"
    "alignas(4) static const unsigned char kArgusAdamwShaderSpv[] = {\n"
)

set(ARGUS_SHADER_OFFSET 0)
while(ARGUS_SHADER_OFFSET LESS ARGUS_SHADER_HEX_LENGTH)
    string(SUBSTRING "${ARGUS_SHADER_HEX}" ${ARGUS_SHADER_OFFSET} 2 ARGUS_SHADER_BYTE)
    file(APPEND "${ARGUS_SHADER_HEADER}" "0x${ARGUS_SHADER_BYTE},")
    math(EXPR ARGUS_SHADER_OFFSET "${ARGUS_SHADER_OFFSET} + 2")
    math(EXPR ARGUS_SHADER_COLUMN "${ARGUS_SHADER_OFFSET} % 32")
    if(ARGUS_SHADER_COLUMN EQUAL 0)
        file(APPEND "${ARGUS_SHADER_HEADER}" "\n")
    endif()
endwhile()

file(
    APPEND "${ARGUS_SHADER_HEADER}"
    "\n};\n"
    "static constexpr std::size_t kArgusAdamwShaderSpvSize = ${ARGUS_SHADER_BYTE_COUNT};\n"
)

add_library(argus_training SHARED vulkan_training_jni.cpp)

target_compile_features(argus_training PRIVATE cxx_std_20)
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
