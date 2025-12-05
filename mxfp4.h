#ifndef MXFP4_H
#define MXFP4_H

#include <cstdint>
#include <cmath>

// MXFP4 format: 4-bit floating point with shared exponent (microscaling)
// Each block of 32 elements shares one 8-bit scale factor (exponent)
// Each element has 1 sign bit and 3 magnitude bits

constexpr int BLOCK_SIZE = 32;  // Number of elements sharing a scale

struct MXFP4Block {
    uint8_t scale;           // Shared 8-bit exponent for the block
    uint8_t data[16];        // 32 x 4-bit values packed into 16 bytes
};

class MXFP4 {
public:
    // Convert float to MXFP4 4-bit representation
    static uint8_t float_to_mxfp4(float value, float scale_factor);
    
    // Convert MXFP4 4-bit representation to float
    static float mxfp4_to_float(uint8_t mxfp4_val, float scale_factor);
    
    // Encode a block of floats to MXFP4
    static void encode_block(const float* input, MXFP4Block& block);
    
    // Decode a block of MXFP4 to floats
    static void decode_block(const MXFP4Block& block, float* output);
    
    // Get 4-bit value from packed data
    static uint8_t get_4bit(const uint8_t* data, int index);
    
    // Set 4-bit value in packed data
    static void set_4bit(uint8_t* data, int index, uint8_t value);
};

#endif // MXFP4_H
