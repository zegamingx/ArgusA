package com.argus.agi.model

/**
 * Compact IEEE-754 binary16 parameter storage.
 *
 * Parameters are stored as 16-bit values, while forward/backward arithmetic,
 * gradients, and AdamW moments remain FP32. This avoids low-precision gradient
 * accumulation and optimizer-state underflow at the cost of quantizing each
 * updated parameter back to binary16.
 */
class Float16ParameterArray(size: Int) {

    private val storage: ShortArray

    init {
        require(size >= 0) { "Parameter-array size cannot be negative." }
        storage = ShortArray(size)
    }

    val size: Int
        get() = storage.size

    val indices: IntRange
        get() = storage.indices

    operator fun get(index: Int): Float = Float16Codec.toFloat(storage[index])

    operator fun set(index: Int, value: Float) {
        require(value.isFinite()) { "FP16 parameter values must remain finite." }
        storage[index] = Float16Codec.fromFloat(value)
    }

    fun fill(value: Float) {
        require(value.isFinite()) { "FP16 parameter values must remain finite." }
        storage.fill(Float16Codec.fromFloat(value))
    }

    fun copyFrom(source: Float16ParameterArray) {
        require(source.size == size) { "FP16 parameter-array sizes must match." }
        System.arraycopy(source.storage, 0, storage, 0, size)
    }

    fun copyFrom(
        source: Float16ParameterArray,
        sourceOffset: Int,
        destinationOffset: Int,
        length: Int
    ) {
        require(sourceOffset >= 0 && destinationOffset >= 0 && length >= 0) {
            "FP16 copy offsets and length must be non-negative."
        }
        require(sourceOffset + length <= source.size) {
            "FP16 source range exceeds its parameter array."
        }
        require(destinationOffset + length <= size) {
            "FP16 destination range exceeds its parameter array."
        }
        System.arraycopy(source.storage, sourceOffset, storage, destinationOffset, length)
    }

    fun toFloatArray(): FloatArray = FloatArray(size) { index -> get(index) }

    companion object {
        fun fromFloatArray(values: FloatArray): Float16ParameterArray =
            Float16ParameterArray(values.size).also { result ->
                for (index in values.indices) {
                    result[index] = values[index]
                }
            }
    }
}

/** Correctly rounded IEEE-754 binary32/binary16 conversion helpers. */
internal object Float16Codec {

    fun fromFloat(value: Float): Short {
        val bits = value.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val absoluteBits = bits and 0x7FFF_FFFF

        if (absoluteBits >= 0x7F80_0000) {
            val payload = if (absoluteBits == 0x7F80_0000) 0 else 0x0200
            return (sign or 0x7C00 or payload).toShort()
        }

        val exponent = ((bits ushr 23) and 0xFF) - 127 + 15
        val mantissa = bits and 0x007F_FFFF

        if (exponent >= 31) {
            return (sign or 0x7C00).toShort()
        }

        if (exponent <= 0) {
            if (exponent < -10) {
                return sign.toShort()
            }

            val significand = mantissa or 0x0080_0000
            val shift = 14 - exponent
            var rounded = significand ushr shift
            val remainderMask = (1 shl shift) - 1
            val remainder = significand and remainderMask
            val halfway = 1 shl (shift - 1)
            if (remainder > halfway || (remainder == halfway && (rounded and 1) != 0)) {
                rounded++
            }
            return (sign or rounded).toShort()
        }

        var halfMantissa = mantissa ushr 13
        val discarded = mantissa and 0x1FFF
        if (discarded > 0x1000 || (discarded == 0x1000 && (halfMantissa and 1) != 0)) {
            halfMantissa++
        }

        var halfExponent = exponent
        if (halfMantissa == 0x0400) {
            halfMantissa = 0
            halfExponent++
            if (halfExponent >= 31) {
                return (sign or 0x7C00).toShort()
            }
        }

        return (sign or (halfExponent shl 10) or halfMantissa).toShort()
    }

    fun toFloat(value: Short): Float {
        val half = value.toInt() and 0xFFFF
        val sign = (half and 0x8000) shl 16
        val exponent = (half ushr 10) and 0x1F
        val mantissa = half and 0x03FF

        val floatBits = when (exponent) {
            0 -> {
                if (mantissa == 0) {
                    sign
                } else {
                    var normalizedMantissa = mantissa
                    var normalizedExponent = -14
                    while ((normalizedMantissa and 0x0400) == 0) {
                        normalizedMantissa = normalizedMantissa shl 1
                        normalizedExponent--
                    }
                    normalizedMantissa = normalizedMantissa and 0x03FF
                    sign or
                        ((normalizedExponent + 127) shl 23) or
                        (normalizedMantissa shl 13)
                }
            }
            0x1F -> {
                sign or 0x7F80_0000 or (mantissa shl 13)
            }
            else -> {
                sign or ((exponent - 15 + 127) shl 23) or (mantissa shl 13)
            }
        }

        return Float.fromBits(floatBits)
    }
}
