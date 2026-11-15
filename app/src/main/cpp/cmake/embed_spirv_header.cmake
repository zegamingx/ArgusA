if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE OR
   NOT DEFINED ARRAY_SYMBOL OR NOT DEFINED SIZE_SYMBOL)
    message(FATAL_ERROR
        "SPIR-V embedding requires INPUT_FILE, OUTPUT_FILE, ARRAY_SYMBOL, and SIZE_SYMBOL."
    )
endif()

if(NOT EXISTS "${INPUT_FILE}")
    message(FATAL_ERROR "SPIR-V compiler did not produce: ${INPUT_FILE}")
endif()

file(SIZE "${INPUT_FILE}" shader_byte_count)
if(shader_byte_count LESS 20)
    message(FATAL_ERROR "Compiled SPIR-V is too small to contain a valid module: ${INPUT_FILE}")
endif()

math(EXPR shader_word_remainder "${shader_byte_count} % 4")
if(NOT shader_word_remainder EQUAL 0)
    message(FATAL_ERROR "Compiled SPIR-V size is not word-aligned: ${INPUT_FILE}")
endif()

file(READ "${INPUT_FILE}" shader_hex HEX)
string(LENGTH "${shader_hex}" shader_hex_length)
math(EXPR expected_hex_length "${shader_byte_count} * 2")
if(NOT shader_hex_length EQUAL expected_hex_length)
    message(FATAL_ERROR "Could not read the complete SPIR-V module: ${INPUT_FILE}")
endif()

string(SUBSTRING "${shader_hex}" 0 8 shader_magic)
if(NOT shader_magic STREQUAL "03022307")
    message(FATAL_ERROR "Compiled shader does not contain the SPIR-V magic number: ${INPUT_FILE}")
endif()

get_filename_component(output_directory "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")

file(
    WRITE "${OUTPUT_FILE}"
    "#pragma once\n"
    "#include <cstddef>\n"
    "#include <cstdint>\n"
    "alignas(4) static const unsigned char ${ARRAY_SYMBOL}[] = {\n"
)

set(shader_offset 0)
while(shader_offset LESS shader_hex_length)
    string(SUBSTRING "${shader_hex}" ${shader_offset} 2 shader_byte)
    file(APPEND "${OUTPUT_FILE}" "0x${shader_byte},")

    math(EXPR shader_offset "${shader_offset} + 2")
    math(EXPR shader_column "${shader_offset} % 32")
    if(shader_column EQUAL 0)
        file(APPEND "${OUTPUT_FILE}" "\n")
    endif()
endwhile()

file(
    APPEND "${OUTPUT_FILE}"
    "\n};\n"
    "static constexpr std::size_t ${SIZE_SYMBOL} = ${shader_byte_count};\n"
)
