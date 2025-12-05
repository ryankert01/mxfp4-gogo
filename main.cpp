#include <iostream>
#include <chrono>
#include <iomanip>
#include "matrix.h"
#include "mxfp4.h"

using namespace std;
using namespace std::chrono;

void test_mxfp4_encoding() {
    cout << "\n=== Testing MXFP4 Encoding/Decoding ===" << endl;
    
    float test_data[BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; i++) {
        test_data[i] = (i - 16) * 0.5f;  // Values from -8 to 7.5
    }
    
    cout << "Original values (first 10): ";
    for (int i = 0; i < 10; i++) {
        cout << test_data[i] << " ";
    }
    cout << endl;
    
    MXFP4Block block;
    MXFP4::encode_block(test_data, block);
    
    float decoded[BLOCK_SIZE];
    MXFP4::decode_block(block, decoded);
    
    cout << "Decoded values (first 10):  ";
    for (int i = 0; i < 10; i++) {
        cout << decoded[i] << " ";
    }
    cout << endl;
    
    // Calculate error
    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float error = abs(test_data[i] - decoded[i]);
        max_error = max(max_error, error);
        avg_error += error;
    }
    avg_error /= BLOCK_SIZE;
    
    cout << "Max error: " << max_error << endl;
    cout << "Avg error: " << avg_error << endl;
    cout << "MXFP4 encoding test: PASSED" << endl;
}

void verify_correctness(int size) {
    cout << "\n=== Verifying Matrix Multiplication Correctness ===" << endl;
    cout << "Matrix size: " << size << "x" << size << endl;
    
    Matrix A(size, size);
    Matrix B(size, size);
    
    A.randomize(-5.0f, 5.0f);
    B.randomize(-5.0f, 5.0f);
    
    cout << "Computing sequential result..." << endl;
    auto C_seq = matmul_sequential(A, B);
    
    cout << "Computing parallel result..." << endl;
    auto C_par = matmul_parallel(A, B, 4);
    
    cout << "Computing optimized result..." << endl;
    auto C_opt = matmul_optimized(A, B, 4);
    
    bool seq_par_match = C_seq.approx_equal(C_par, 0.5f);
    bool seq_opt_match = C_seq.approx_equal(C_opt, 0.5f);
    
    cout << "Sequential vs Parallel: " << (seq_par_match ? "MATCH" : "MISMATCH") << endl;
    cout << "Sequential vs Optimized: " << (seq_opt_match ? "MATCH" : "MISMATCH") << endl;
    
    if (seq_par_match && seq_opt_match) {
        cout << "Verification: PASSED" << endl;
    } else {
        cout << "Verification: FAILED" << endl;
        cout << "\nSample values from result matrices:" << endl;
        cout << "Sequential:" << endl;
        C_seq.print();
        cout << "Parallel:" << endl;
        C_par.print();
        cout << "Optimized:" << endl;
        C_opt.print();
    }
}

void benchmark(int size, int iterations = 3) {
    cout << "\n=== Benchmarking Matrix Multiplication ===" << endl;
    cout << "Matrix size: " << size << "x" << size << endl;
    cout << "Iterations: " << iterations << endl;
    
    Matrix A(size, size);
    Matrix B(size, size);
    
    A.randomize(-5.0f, 5.0f);
    B.randomize(-5.0f, 5.0f);
    
    // Benchmark sequential
    double seq_total = 0.0;
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        auto C = matmul_sequential(A, B);
        auto end = high_resolution_clock::now();
        seq_total += duration_cast<milliseconds>(end - start).count();
    }
    double seq_avg = seq_total / iterations;
    
    // Benchmark parallel (4 threads)
    double par_total = 0.0;
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        auto C = matmul_parallel(A, B, 4);
        auto end = high_resolution_clock::now();
        par_total += duration_cast<milliseconds>(end - start).count();
    }
    double par_avg = par_total / iterations;
    
    // Benchmark optimized (4 threads)
    double opt_total = 0.0;
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        auto C = matmul_optimized(A, B, 4);
        auto end = high_resolution_clock::now();
        opt_total += duration_cast<milliseconds>(end - start).count();
    }
    double opt_avg = opt_total / iterations;
    
    cout << "\nResults:" << endl;
    cout << fixed << setprecision(2);
    cout << "Sequential:  " << seq_avg << " ms" << endl;
    cout << "Parallel:    " << par_avg << " ms (speedup: " << seq_avg/par_avg << "x)" << endl;
    cout << "Optimized:   " << opt_avg << " ms (speedup: " << seq_avg/opt_avg << "x)" << endl;
}

int main(int argc, char** argv) {
    cout << "MXFP4 Matrix Multiplication Implementation" << endl;
    cout << "===========================================" << endl;
    
    // Test MXFP4 encoding/decoding
    test_mxfp4_encoding();
    
    // Verify correctness with small matrix
    verify_correctness(64);
    
    // Benchmark with different sizes
    cout << "\n\n";
    benchmark(128, 3);
    
    cout << "\n";
    benchmark(256, 3);
    
    cout << "\n";
    benchmark(512, 3);
    
    return 0;
}
