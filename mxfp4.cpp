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
    
    // Store scale_factor using a simple quantization scheme
    // Scale to 0-255 range for 8-bit storage
    float log_scale = (scale_factor > 0) ? std::log2(scale_factor) : -10.0f;
    // Map to 8-bit range: log scale typically in range [-10, 10]
    int quantized_scale = static_cast<int>((log_scale + 10.0f) * 12.75f);
    quantized_scale = std::min(255, std::max(0, quantized_scale));
    block.scale = static_cast<uint8_t>(quantized_scale);
    
    // Initialize data
    memset(block.data, 0, sizeof(block.data));
    
    // Encode each value
    for (int i = 0; i < BLOCK_SIZE; i++) {
        uint8_t encoded = float_to_mxfp4(input[i], scale_factor);
        set_4bit(block.data, i, encoded);
    }
}

void MXFP4::decode_block(const MXFP4Block& block, float* output) {
    // Reconstruct scale factor from stored value
    float log_scale = (static_cast<float>(block.scale) / 12.75f) - 10.0f;
    float scale_factor = std::exp2(log_scale);
    
    // Decode each value
    for (int i = 0; i < BLOCK_SIZE; i++) {
        uint8_t encoded = get_4bit(block.data, i);
        output[i] = mxfp4_to_float(encoded, scale_factor);
    }
}
