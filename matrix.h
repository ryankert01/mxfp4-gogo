#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include "mxfp4.h"

class Matrix {
public:
    int rows;
    int cols;
    std::vector<float> data;
    
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
    
    float& operator()(int i, int j) {
        return data[i * cols + j];
    }
    
    const float& operator()(int i, int j) const {
        return data[i * cols + j];
    }
    
    void randomize(float min = -1.0f, float max = 1.0f);
    void print() const;
    bool approx_equal(const Matrix& other, float tolerance = 0.1f) const;
};

// Sequential matrix multiplication using MXFP4
Matrix matmul_sequential(const Matrix& A, const Matrix& B);

// Parallel matrix multiplication using pthread
Matrix matmul_parallel(const Matrix& A, const Matrix& B, int num_threads = 4);

// Optimized parallel version with cacheline and SIMD
Matrix matmul_optimized(const Matrix& A, const Matrix& B, int num_threads = 4);

#endif // MATRIX_H
