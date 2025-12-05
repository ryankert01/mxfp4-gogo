#include "mxfp4.h"
#include <algorithm>
#include <cstring>

uint8_t MXFP4::get_4bit(const uint8_t* data, int index) {
    int byte_idx = index / 2;
    if (index % 2 == 0) {
        return data[byte_idx] & 0x0F;
    } else {
        return (data[byte_idx] >> 4) & 0x0F;
    }
}

void MXFP4::set_4bit(uint8_t* data, int index, uint8_t value) {
    int byte_idx = index / 2;
    value &= 0x0F;  // Ensure only 4 bits
    if (index % 2 == 0) {
        data[byte_idx] = (data[byte_idx] & 0xF0) | value;
    } else {
        data[byte_idx] = (data[byte_idx] & 0x0F) | (value << 4);
    }
}

uint8_t MXFP4::float_to_mxfp4(float value, float scale_factor) {
    if (value == 0.0f) return 0;
    
    // Get sign bit
    uint8_t sign = (value < 0) ? 0x8 : 0x0;
    float abs_val = std::abs(value);
    
    // Scale the value
    float scaled = abs_val / scale_factor;
    
    // Clamp to representable range [0, 7] for 3-bit magnitude
    int magnitude = std::min(7, std::max(0, static_cast<int>(std::round(scaled))));
    
    return sign | magnitude;
}

float MXFP4::mxfp4_to_float(uint8_t mxfp4_val, float scale_factor) {
    // Extract sign and magnitude
    bool is_negative = (mxfp4_val & 0x8) != 0;
    int magnitude = mxfp4_val & 0x7;
    
    float value = magnitude * scale_factor;
    return is_negative ? -value : value;
}

void MXFP4::encode_block(const float* input, MXFP4Block& block) {
    // Find maximum absolute value in the block to determine scale
    float max_abs = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        max_abs = std::max(max_abs, std::abs(input[i]));
    }
    
    // Calculate scale factor
    // We have 3 bits for magnitude (0-7), so divide max by 7
    float scale_factor = (max_abs > 0) ? (max_abs / 7.0f) : 1.0f;
    
    // Store scale as 8-bit exponent (simplified: just use bits of float)
    // For simplicity, we'll store the scale_factor directly as a packed representation
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale_factor, sizeof(float));
    block.scale = (scale_bits >> 23) & 0xFF;  // Use exponent bits
    
    // Initialize data
    memset(block.data, 0, sizeof(block.data));
    
    // Encode each value
    for (int i = 0; i < BLOCK_SIZE; i++) {
        uint8_t encoded = float_to_mxfp4(input[i], scale_factor);
        set_4bit(block.data, i, encoded);
    }
}

void MXFP4::decode_block(const MXFP4Block& block, float* output) {
    // Reconstruct scale factor from stored exponent
    uint32_t scale_bits = (static_cast<uint32_t>(block.scale) << 23) | 0x3F800000;
    float scale_factor;
    memcpy(&scale_factor, &scale_bits, sizeof(float));
    scale_factor = scale_factor - 1.0f;  // Adjust for bias
    if (scale_factor <= 0) scale_factor = 1.0f;
    
    // Decode each value
    for (int i = 0; i < BLOCK_SIZE; i++) {
        uint8_t encoded = get_4bit(block.data, i);
        output[i] = mxfp4_to_float(encoded, scale_factor);
    }
}
