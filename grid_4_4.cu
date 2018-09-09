#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gputimer.h"

#define grid_size 4
#define total_entries (grid_size * grid_size)

#define top 0
#define right 1
#define bottom 2
#define left 3

#define rho 0.5
#define eta 2E-4
#define G 0.75

struct element_s {
    float u;
    float u1;
    float u2;
};

// Create a grid
int createMatrix(struct element_s *matrix, int hit) {
    //struct element_s matrix[total_entries] = {0};
    for (int i = 0; i < total_entries; i++) {
        matrix[i].u = 0;
        matrix[i].u1 = 0;
        matrix[i].u2 = 0;
    }
    if (hit == 1) {
        int hit_idx = grid_size * (grid_size / 2) + (grid_size / 2);
        matrix[hit_idx].u = 1;
        matrix[hit_idx].u1 = 1;
    }
    return 0;
}
__device__ int get_array_idx(int i, int j) {
    return (grid_size * i + j);
}

__device__ int idx_to_ij(int idx, int position) {
    int i = idx / grid_size;
    int j = idx % grid_size;

    switch(position){
        case 0:
            return (grid_size * (i - 1) + j);

        case 1:
            return (grid_size * i + (j + 1));

        case 2:
            return (grid_size * (i + 1) + j);

        case 3:
            return (grid_size * i + (j - 1));

        default:
            return 0;
    }
}

__device__ __host__ int store_prev(struct element_s *entry, float new_value) {
    entry->u2 = entry->u1;
    entry->u1 = entry->u;
    entry->u = new_value;

    return 0;
}

int printMatrix(struct element_s *matrix, int u_idx) {
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
        	if (u_idx == 0) {
        		printf("(%d, %d) = %.6f\t", i, j, matrix[grid_size * i + j].u);
        	}
            else if (u_idx == 1) {
            	printf("(%d, %d) = %.6f\t", i, j, matrix[grid_size * i + j].u1);
            }
            else if (u_idx == 2) {
            	printf("(%d, %d) = %.6f\t", i, j, matrix[grid_size * i + j].u2);
            }
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}

int printOutput(struct element_s *matrix) {
    int output_idx = grid_size * (grid_size / 2) + (grid_size / 2);
    //printf("(%d, %d) %.6f\n", grid_size / 2, grid_size / 2, matrix[output_idx].u);
    printf("%.6f\n", matrix[output_idx].u);

    return 0;
}

__device__ int printLocal(struct element_s element) {
    printf("u = %.6f\t", element.u);
    printf("u1 = %.6f\t", element.u1);
    printf("u2 = %.6f\n", element.u2);

    return 0;
}
__global__ void interior(struct element_s * d_out, struct element_s * d_in ){
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    //printf("interior: thread (%d,%d) in block (%d,%d): idx = %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, idx);
        
    if (!(idx % grid_size == 0 || idx % grid_size == (grid_size - 1) ||idx < get_array_idx(0, grid_size - 1) || idx > get_array_idx(grid_size - 2, grid_size - 1))){
        //interior elements
        float temp;
        temp = (rho*(d_in[idx_to_ij(idx, top)].u1 + d_in[idx_to_ij(idx, bottom)].u1 + d_in[idx_to_ij(idx, left)].u1 + d_in[idx_to_ij(idx, right)].u1 - 4 * d_in[idx].u1)
                    + 2 * d_in[idx].u1 - (1 - eta) * d_in[idx].u2) / (1 + eta);
        store_prev(&d_in[idx], temp);
        printf("interior idx:%d, d_in[idx].u:%.6f\n",idx,d_in[idx].u);
        
    }
    d_out[idx] = d_in[idx];
    
}
__global__ void edge(struct element_s * d_out, struct element_s * d_in){
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    //printf("edge: thread (%d,%d) in block (%d,%d): idx = %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, idx);
        
    if (idx == get_array_idx(0, 0) || idx == get_array_idx(0, grid_size - 1) ||
                idx == get_array_idx(grid_size - 1, 0) || idx == get_array_idx(grid_size - 1, grid_size - 1)) {
        //ignore elements located at the corner
    }
    //top edge
    else if (idx < get_array_idx(0, grid_size)) {
        float temp;
        temp = G * d_in[idx + grid_size].u;
        store_prev(&d_in[idx], temp);
        printf("edge idx:%d, d_in[idx].u:%.6f\n",idx,d_in[idx].u);
    }
    //bottom edge
    else if (idx > get_array_idx(grid_size - 2, grid_size - 1)) {
        float temp;
        temp = G * d_in[idx - grid_size].u;
        store_prev(&d_in[idx], temp);
        printf("edge idx:%d, d_in[idx].u:%.6f\n",idx,d_in[idx].u);
    }
    //left edge
    else if (idx % grid_size == 0) {
        float temp;
        temp = G * d_in[idx + 1].u;
        store_prev(&d_in[idx], temp);
        printf("edge idx:%d, d_in[idx].u:%.6f\n",idx,d_in[idx].u);
    }
    //right edge
    else if (idx % grid_size == (grid_size - 1)) {
        float temp;
        temp = G * d_in[idx - 1].u;
        store_prev(&d_in[idx], temp);
        printf("edge idx:%d, d_in[idx].u:%.6f\n",idx,d_in[idx].u);
    }

    d_out[idx] = d_in[idx];
}
__global__ void corner(struct element_s * d_out, struct element_s * d_in){
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    //printf("corner: thread (%d,%d) in block (%d,%d): idx = %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, idx);

    //top left corner
    if (idx == get_array_idx(0, 0)) {
        float temp;
        temp = G * d_in[idx + grid_size].u;
        store_prev(&d_in[idx], temp);
        //printf("corner: thread (%d,%d) in block (%d,%d): idx = %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, idx);
        printf("corner idx:%d, d_in[idx].u:%.6f\n",idx,d_in[idx].u);
    }
    //top right and bottom right
    else if (idx == get_array_idx(0, grid_size - 1) || idx == get_array_idx(grid_size - 1, grid_size - 1)) {
        float temp;
        temp = G * d_in[idx - 1].u;
        store_prev(&d_in[idx], temp);
        //printf("corner: thread (%d,%d) in block (%d,%d): idx = %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, idx);
        printf("corner idx:%d, d_in[idx].u:%.6f\n",idx,d_in[idx].u);
    }
    //bottom left
    else if (idx == get_array_idx(grid_size - 1, 0)) {
        float temp;
        temp = G * d_in[idx - grid_size].u;
        store_prev(&d_in[idx], temp);
        //printf("corner: thread (%d,%d) in block (%d,%d): idx = %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, idx);
        printf("corner idx:%d, d_in[idx].u:%.6f\n",idx,d_in[idx].u);
    }
    
    d_out[idx] = d_in[idx];
}
int main(int argc, char **argv) {
    const int ARRAY_BYTES = total_entries * sizeof(element_s);

	if(argv[1] == NULL)
    {
        printf("Missing argument for iteration\n");
        return 0;
    }
    int T = atoi(argv[1]);
    // initialize the input array on the host
    struct element_s h_array[total_entries] = {0};
    createMatrix(h_array, 1);   

    // declare GPU memory pointers
    struct element_s * d_in;
    struct element_s * d_out;

    // allocate GPU memory
    cudaMalloc(&d_in, ARRAY_BYTES);
    cudaMalloc(&d_out, ARRAY_BYTES);

    GpuTimer timer;

    dim3 dimBlock(grid_size, grid_size, 1);
    dim3 dimGrid(1, 1, 1);
    while (T --> 0) {
        //interior
        // transfer the array to the GPU
        cudaMemcpy(d_in, h_array, ARRAY_BYTES, cudaMemcpyHostToDevice);
        // launch the kernel to update interior elements
        timer.Start();
        interior<<<dimGrid, dimBlock>>>(d_out, d_in);
        timer.Stop();
        // copy back the result array to the CPU
        cudaMemcpy(h_array, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost); 

        //edge
        // transfer the updated array to the GPU
        cudaMemcpy(d_in, h_array, ARRAY_BYTES, cudaMemcpyHostToDevice);
        // launch the kernel
        timer.Start();
        edge<<<dimGrid, dimBlock>>>(d_out, d_in);
        timer.Stop();
        // copy back the result array to the CPU
        cudaMemcpy(h_array, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost); 

        //corner
        // transfer the updated array to the GPU
        cudaMemcpy(d_in, h_array, ARRAY_BYTES, cudaMemcpyHostToDevice);
        // launch the kernel
        timer.Start();
        corner<<<dimGrid, dimBlock>>>(d_out, d_in);
        timer.Stop();
        // copy back the result array to the CPU
        cudaMemcpy(h_array, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost); 
        
        for (int i = 0; i < total_entries; i++){
            store_prev(&h_array[i], h_array[i].u);
        }  

        printMatrix(h_array, 0);
        printOutput(h_array);

    }
    printf("Time elapsed = %g ms\n", timer.Elapsed());
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

