#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to apply a 2x2 gate (Umatrix) on specific qubit using shared memory
__global__ void quamsim_kernel(const float *input, float *output, const float *Umatrix, int size, int qbit)
{
    // Allocate shared memory for the input values for faster access.
    extern __shared__ float shared_input[];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Load from global to shared memory
    if (global_idx < size) {
        shared_input[tid] = input[global_idx];
    }
    __syncthreads();

    int mask = 1 << qbit;
    int partner_idx = global_idx ^ mask;

    if (global_idx <= size - mask) {
        if ((global_idx / mask) % 2 != 1) {
            float val_i = (global_idx < size) ? shared_input[tid] : 0.0f;

            float val_partner;
            int partner_tid = partner_idx - blockIdx.x * blockDim.x;

            // Read partner from shared if in same block, else from global
			if (partner_tid >= 0 && partner_tid < blockDim.x) {
                val_partner = shared_input[partner_tid];
            } else {
                val_partner = input[partner_idx];
            }

            output[global_idx] = Umatrix[0] * val_i + Umatrix[1] * val_partner;
            output[partner_idx]  = Umatrix[2] * val_i + Umatrix[3] * val_partner;
        }
    }
}

int main(int argc, char *argv[]){
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <trace_file>" << endl;
        return 1;
    }

    ifstream file(argv[1]);
    if (!file.is_open()) {
        cerr << "Error opening file: " << argv[1] << endl;
        return 1;
    }

    // Read 6 quantum gate matrices (2x2)
	float matrices[6][4];
    for (int m = 0; m < 6; ++m) {
        for (int i = 0; i < 4; ++i) {
            file >> matrices[m][i];
        }
    }

    // Read the quantum state vector
	vector<float> raw_data;
    float temp;
    while (file >> temp) {
        raw_data.push_back(temp);
    }

    // Extract the last 6 values as qubit indices
	float qbits[6];
    for (int i = 5; i >= 0; --i) {
        qbits[i] = raw_data.back();
        raw_data.pop_back();
    }

    int num_elements = raw_data.size();
    size_t vec_size = num_elements * sizeof(float);
    size_t matrix_size = 4 * sizeof(float);

    // Host allocations
	float *h_input = (float *)malloc(vec_size);
    float *h_output = (float *)malloc(vec_size);
    float *h_U = (float *)malloc(matrix_size);

    for (int i = 0; i < num_elements; ++i)
        h_input[i] = raw_data[i];

    float *d_input, *d_output, *d_U;
    // Device allocations
	cudaMalloc(&d_input, vec_size);
    cudaMalloc(&d_output, vec_size);
    cudaMalloc(&d_U, matrix_size);

    cudaMemcpy(d_input, h_input, vec_size, cudaMemcpyHostToDevice);

    int threads_per_block = 64;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Apply 6 gates in sequence
	for (int stage = 0; stage < 6; ++stage) {
        for (int i = 0; i < 4; ++i)
            h_U[i] = matrices[stage][i];

        cudaMemcpy(d_U, h_U, matrix_size, cudaMemcpyHostToDevice);

        quamsim_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, d_U, num_elements, qbits[stage]);
        cudaDeviceSynchronize();

        cudaMemcpy(h_input, d_output, vec_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_input, h_input, vec_size, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(h_output, d_output, vec_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_elements; ++i)
        printf("%.3f\n", h_output[i]);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_U);
    free(h_input);
    free(h_output);
    free(h_U);
    
    cudaDeviceReset();

    return 0;
}
