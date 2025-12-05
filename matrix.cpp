#include "matrix.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <pthread.h>
#include <immintrin.h>
#include <algorithm>

void Matrix::randomize(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (auto& val : data) {
        val = dis(gen);
    }
}

void Matrix::print() const {
    for (int i = 0; i < std::min(rows, 10); i++) {
        for (int j = 0; j < std::min(cols, 10); j++) {
            std::cout << std::setw(8) << std::setprecision(3) << (*this)(i, j) << " ";
        }
        if (cols > 10) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 10) std::cout << "..." << std::endl;
}

bool Matrix::approx_equal(const Matrix& other, float tolerance) const {
    if (rows != other.rows || cols != other.cols) return false;
    
    for (size_t i = 0; i < data.size(); i++) {
        if (std::abs(data[i] - other.data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Sequential MXFP4 matrix multiplication
Matrix matmul_sequential(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }
    
    Matrix C(A.rows, B.cols);
    
    // Process matrices in blocks for MXFP4 encoding
    int num_blocks_A = (A.rows * A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_B = (B.rows * B.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    std::vector<MXFP4Block> blocks_A(num_blocks_A);
    std::vector<MXFP4Block> blocks_B(num_blocks_B);
    
    // Encode matrices to MXFP4
    for (int b = 0; b < num_blocks_A; b++) {
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(A.data.size()));
        
        float block_data[BLOCK_SIZE] = {0};
        for (int i = start_idx; i < end_idx; i++) {
            block_data[i - start_idx] = A.data[i];
        }
        MXFP4::encode_block(block_data, blocks_A[b]);
    }
    
    for (int b = 0; b < num_blocks_B; b++) {
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(B.data.size()));
        
        float block_data[BLOCK_SIZE] = {0};
        for (int i = start_idx; i < end_idx; i++) {
            block_data[i - start_idx] = B.data[i];
        }
        MXFP4::encode_block(block_data, blocks_B[b]);
    }
    
    // Decode back for computation (in real scenario, computation would be in MXFP4)
    std::vector<float> A_decoded(A.data.size());
    std::vector<float> B_decoded(B.data.size());
    
    for (int b = 0; b < num_blocks_A; b++) {
        float block_data[BLOCK_SIZE];
        MXFP4::decode_block(blocks_A[b], block_data);
        
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(A.data.size()));
        for (int i = start_idx; i < end_idx; i++) {
            A_decoded[i] = block_data[i - start_idx];
        }
    }
    
    for (int b = 0; b < num_blocks_B; b++) {
        float block_data[BLOCK_SIZE];
        MXFP4::decode_block(blocks_B[b], block_data);
        
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(B.data.size()));
        for (int i = start_idx; i < end_idx; i++) {
            B_decoded[i] = block_data[i - start_idx];
        }
    }
    
    // Perform matrix multiplication
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; k++) {
                sum += A_decoded[i * A.cols + k] * B_decoded[k * B.cols + j];
            }
            C(i, j) = sum;
        }
    }
    
    return C;
}

// Pthread data structure
struct ThreadData {
    const std::vector<float>* A_decoded;
    const std::vector<float>* B_decoded;
    const std::vector<float>* B_transposed;  // Added for optimized version
    Matrix* C;
    int A_rows;
    int A_cols;
    int B_cols;
    int start_row;
    int end_row;
};

void* matmul_thread_worker(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->B_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < data->A_cols; k++) {
                sum += (*data->A_decoded)[i * data->A_cols + k] * 
                       (*data->B_decoded)[k * data->B_cols + j];
            }
            (*data->C)(i, j) = sum;
        }
    }
    
    return nullptr;
}

// Parallel matrix multiplication with pthread
Matrix matmul_parallel(const Matrix& A, const Matrix& B, int num_threads) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }
    
    Matrix C(A.rows, B.cols);
    
    // Encode to MXFP4
    int num_blocks_A = (A.rows * A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_B = (B.rows * B.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    std::vector<MXFP4Block> blocks_A(num_blocks_A);
    std::vector<MXFP4Block> blocks_B(num_blocks_B);
    
    for (int b = 0; b < num_blocks_A; b++) {
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(A.data.size()));
        
        float block_data[BLOCK_SIZE] = {0};
        for (int i = start_idx; i < end_idx; i++) {
            block_data[i - start_idx] = A.data[i];
        }
        MXFP4::encode_block(block_data, blocks_A[b]);
    }
    
    for (int b = 0; b < num_blocks_B; b++) {
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(B.data.size()));
        
        float block_data[BLOCK_SIZE] = {0};
        for (int i = start_idx; i < end_idx; i++) {
            block_data[i - start_idx] = B.data[i];
        }
        MXFP4::encode_block(block_data, blocks_B[b]);
    }
    
    // Decode
    std::vector<float> A_decoded(A.data.size());
    std::vector<float> B_decoded(B.data.size());
    
    for (int b = 0; b < num_blocks_A; b++) {
        float block_data[BLOCK_SIZE];
        MXFP4::decode_block(blocks_A[b], block_data);
        
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(A.data.size()));
        for (int i = start_idx; i < end_idx; i++) {
            A_decoded[i] = block_data[i - start_idx];
        }
    }
    
    for (int b = 0; b < num_blocks_B; b++) {
        float block_data[BLOCK_SIZE];
        MXFP4::decode_block(blocks_B[b], block_data);
        
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(B.data.size()));
        for (int i = start_idx; i < end_idx; i++) {
            B_decoded[i] = block_data[i - start_idx];
        }
    }
    
    // Create threads
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);
    
    int rows_per_thread = A.rows / num_threads;
    int remaining_rows = A.rows % num_threads;
    
    int current_row = 0;
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].A_decoded = &A_decoded;
        thread_data[t].B_decoded = &B_decoded;
        thread_data[t].C = &C;
        thread_data[t].A_rows = A.rows;
        thread_data[t].A_cols = A.cols;
        thread_data[t].B_cols = B.cols;
        thread_data[t].start_row = current_row;
        thread_data[t].end_row = current_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
        
        pthread_create(&threads[t], nullptr, matmul_thread_worker, &thread_data[t]);
        
        current_row = thread_data[t].end_row;
    }
    
    // Join threads
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
    }
    
    return C;
}

// Optimized worker with cacheline and SIMD
void* matmul_optimized_worker(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    
    // Using transposed B for better cache locality
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->B_cols; j++) {
            float sum = 0.0f;
            
            int k = 0;
            
#ifdef __AVX__
            // Process 8 floats at a time with AVX
            // Now both A[i,:] and B_transposed[j,:] are contiguous
            __m256 sum_vec = _mm256_setzero_ps();
            for (; k + 7 < data->A_cols; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&(*data->A_decoded)[i * data->A_cols + k]);
                __m256 b_vec = _mm256_loadu_ps(&(*data->B_transposed)[j * data->A_cols + k]);
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            
            // Horizontal sum using hadd
            __m256 hsum = _mm256_hadd_ps(sum_vec, sum_vec);
            hsum = _mm256_hadd_ps(hsum, hsum);
            // Add upper and lower 128-bit lanes
            __m128 sum_high = _mm256_extractf128_ps(hsum, 1);
            __m128 sum_low = _mm256_castps256_ps128(hsum);
            __m128 sum_result = _mm_add_ps(sum_low, sum_high);
            sum += _mm_cvtss_f32(sum_result);
#endif
            
            // Process remaining elements
            for (; k < data->A_cols; k++) {
                sum += (*data->A_decoded)[i * data->A_cols + k] * 
                       (*data->B_transposed)[j * data->A_cols + k];
            }
            
            (*data->C)(i, j) = sum;
        }
    }
    
    return nullptr;
}

// Optimized parallel matrix multiplication
Matrix matmul_optimized(const Matrix& A, const Matrix& B, int num_threads) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }
    
    Matrix C(A.rows, B.cols);
    
    // Encode to MXFP4
    int num_blocks_A = (A.rows * A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_B = (B.rows * B.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    std::vector<MXFP4Block> blocks_A(num_blocks_A);
    std::vector<MXFP4Block> blocks_B(num_blocks_B);
    
    for (int b = 0; b < num_blocks_A; b++) {
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(A.data.size()));
        
        float block_data[BLOCK_SIZE] = {0};
        for (int i = start_idx; i < end_idx; i++) {
            block_data[i - start_idx] = A.data[i];
        }
        MXFP4::encode_block(block_data, blocks_A[b]);
    }
    
    for (int b = 0; b < num_blocks_B; b++) {
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(B.data.size()));
        
        float block_data[BLOCK_SIZE] = {0};
        for (int i = start_idx; i < end_idx; i++) {
            block_data[i - start_idx] = B.data[i];
        }
        MXFP4::encode_block(block_data, blocks_B[b]);
    }
    
    // Decode - aligned for better cache performance
    std::vector<float> A_decoded(A.data.size());
    std::vector<float> B_decoded(B.data.size());
    
    for (int b = 0; b < num_blocks_A; b++) {
        float block_data[BLOCK_SIZE];
        MXFP4::decode_block(blocks_A[b], block_data);
        
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(A.data.size()));
        for (int i = start_idx; i < end_idx; i++) {
            A_decoded[i] = block_data[i - start_idx];
        }
    }
    
    for (int b = 0; b < num_blocks_B; b++) {
        float block_data[BLOCK_SIZE];
        MXFP4::decode_block(blocks_B[b], block_data);
        
        int start_idx = b * BLOCK_SIZE;
        int end_idx = std::min(start_idx + BLOCK_SIZE, static_cast<int>(B.data.size()));
        for (int i = start_idx; i < end_idx; i++) {
            B_decoded[i] = block_data[i - start_idx];
        }
    }
    
    // Transpose B for better cache locality with SIMD
    std::vector<float> B_transposed(B.rows * B.cols);
    for (int i = 0; i < B.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            B_transposed[j * B.rows + i] = B_decoded[i * B.cols + j];
        }
    }
    
    // Create threads
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);
    
    int rows_per_thread = A.rows / num_threads;
    int remaining_rows = A.rows % num_threads;
    
    int current_row = 0;
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].A_decoded = &A_decoded;
        thread_data[t].B_decoded = &B_decoded;
        thread_data[t].B_transposed = &B_transposed;
        thread_data[t].C = &C;
        thread_data[t].A_rows = A.rows;
        thread_data[t].A_cols = A.cols;
        thread_data[t].B_cols = B.cols;
        thread_data[t].start_row = current_row;
        thread_data[t].end_row = current_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
        
        pthread_create(&threads[t], nullptr, matmul_optimized_worker, &thread_data[t]);
        
        current_row = thread_data[t].end_row;
    }
    
    // Join threads
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
    }
    
    return C;
}
