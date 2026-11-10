cmake_minimum_required(VERSION 3.22.1)

project(argus_training LANGUAGES CXX)

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

function(argus_compile_shader source_file binary_name header_name array_symbol size_symbol)
    set(shader_binary "${CMAKE_CURRENT_BINARY_DIR}/${binary_name}")
    set(shader_header "${CMAKE_CURRENT_BINARY_DIR}/${header_name}")

    set_property(
        DIRECTORY APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_file}"
    )

    execute_process(
        COMMAND
            "${ARGUS_GLSLC}"
            --target-env=vulkan1.0
            -O
            "${source_file}"
            -o "${shader_binary}"
        RESULT_VARIABLE shader_result
        OUTPUT_VARIABLE shader_stdout
        ERROR_VARIABLE shader_stderr
    )

    if(NOT shader_result EQUAL 0)
        message(FATAL_ERROR
            "Vulkan shader compilation failed for ${source_file}.\n"
            "stdout: ${shader_stdout}\n"
            "stderr: ${shader_stderr}"
        )
    endif()

    file(READ "${shader_binary}" shader_hex HEX)
    string(LENGTH "${shader_hex}" shader_hex_length)
    math(EXPR shader_byte_count "${shader_hex_length} / 2")
    math(EXPR shader_word_remainder "${shader_byte_count} % 4")

    if(shader_byte_count LESS 20 OR NOT shader_word_remainder EQUAL 0)
        message(FATAL_ERROR "Compiled SPIR-V is incomplete or not word-aligned: ${source_file}")
    endif()

    file(
        WRITE "${shader_header}"
        "#pragma once\n"
        "#include <cstddef>\n"
        "#include <cstdint>\n"
        "alignas(4) static const unsigned char ${array_symbol}[] = {\n"
    )

    set(shader_offset 0)
    while(shader_offset LESS shader_hex_length)
        string(SUBSTRING "${shader_hex}" ${shader_offset} 2 shader_byte)
        file(APPEND "${shader_header}" "0x${shader_byte},")
        math(EXPR shader_offset "${shader_offset} + 2")
        math(EXPR shader_column "${shader_offset} % 32")
        if(shader_column EQUAL 0)
            file(APPEND "${shader_header}" "\n")
        endif()
    endwhile()

    file(
        APPEND "${shader_header}"
        "\n};\n"
        "static constexpr std::size_t ${size_symbol} = ${shader_byte_count};\n"
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
