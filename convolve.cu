#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lodepng.h"
#include "gputimer.h"
#include "wm.h"

#define BYTES_PER_PIXEL 4
#define MATRICES_SIZE 3

__host__ __device__ int idx_to_ij(int idx, int ii, int jj, unsigned width) {
    int i, j, position;
    i = (idx / BYTES_PER_PIXEL) / width;
    j = (idx / BYTES_PER_PIXEL) % width;
    position = MATRICES_SIZE * ii + jj;

    switch(position){
        case 0:
            return (width * (i - 1) + (j - 1));
        case 1:
            return (width * (i - 1) + j);
        case 2:
            return (width * (i - 1) + (j + 1));
        case 3:
            return (width * i + (j - 1));
        case 4:
            return (width * i + j);
        case 5:
            return (width * i + (j + 1));
        case 6:
            return (width * (i + 1) + (j - 1));
        case 7:
            return (width * (i + 1) + j);
        case 8:
            return (width * (i + 1) + (j + 1));
        default:
            return 0;
    }
}

__host__ __device__ int convert_to_actual_idx(int idx, unsigned width, unsigned new_width) {
    int i, j;
    i = (idx / BYTES_PER_PIXEL) / new_width;
    j = (idx / BYTES_PER_PIXEL) % new_width;

    return(width * (i + 1) + (j + 1));
}

__global__ void convolve(unsigned char *d_new_image, unsigned char *d_image, float *d_weights, unsigned width, unsigned height, unsigned new_width) {
    int new_idx = BYTES_PER_PIXEL * (blockDim.x * blockIdx.x + threadIdx.x);
    int actual_idx = BYTES_PER_PIXEL * convert_to_actual_idx(new_idx, width, new_width);
    float sum_r, sum_g, sum_b;
    sum_r = 0.0;
    sum_g = 0.0;
    sum_b = 0.0;
    int ii, jj;
    for (ii = 0; ii < MATRICES_SIZE; ii++) {
        for (jj = 0; jj < MATRICES_SIZE; jj++) {
            int idx;
            idx = BYTES_PER_PIXEL * idx_to_ij(actual_idx, ii, jj, width);
            sum_r += d_image[idx] * d_weights[MATRICES_SIZE * ii + jj];
            sum_g += d_image[idx + 1] * d_weights[MATRICES_SIZE * ii + jj];
            sum_b += d_image[idx + 2] * d_weights[MATRICES_SIZE * ii + jj];
        }
    }
    //clamp for R
    sum_r = (sum_r < 0) ? 0 : sum_r;
    sum_r = (sum_r > 255) ? 255 : sum_r;
    //clamp for G
    sum_g = (sum_g < 0) ? 0 : sum_g;
    sum_g = (sum_g > 255) ? 255 : sum_g;
    //clamp for B
    sum_b = (sum_b < 0) ? 0 : sum_b;
    sum_b = (sum_b > 255) ? 255 : sum_b;

    d_new_image[new_idx] = (unsigned char)sum_r;
    d_new_image[new_idx + 1] = (unsigned char)sum_g;
    d_new_image[new_idx + 2] = (unsigned char)sum_b;
    d_new_image[new_idx + 3] = 255;
}

int main(int argc, char **argv) {
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    printf("convolving %s...\n", input_filename);

    unsigned error;
    unsigned char *h_image, *h_new_image;
    unsigned width, height;

    GpuTimer timer;

    error = lodepng_decode32_file(&h_image, &width, &height, input_filename);
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

    const int N = width * height;
    const int TOTAL_BYTES = N * BYTES_PER_PIXEL * sizeof(unsigned char);

    unsigned new_width = width - 2;
    unsigned new_height = height - 2;
    const int new_N = new_width * new_height;
    const int NEW_TOTAL_BYTES = new_N * BYTES_PER_PIXEL * sizeof(unsigned char);

    h_new_image = (unsigned char *) malloc(TOTAL_BYTES);

    float *d_weights;
    const int W_SIZE = 9 * sizeof(float);

    unsigned char *d_image;
    unsigned char *d_new_image;

    cudaMalloc(&d_weights, W_SIZE);
    cudaMalloc(&d_image, TOTAL_BYTES);
    cudaMalloc(&d_new_image, NEW_TOTAL_BYTES);

    cudaMemcpy(d_weights, w, W_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, h_image, TOTAL_BYTES, cudaMemcpyHostToDevice);

    int block_size = 1024;
    int chunk_size = new_N / block_size;
    timer.Start();
    convolve<<<chunk_size, block_size>>>(d_new_image, d_image, d_weights, width, height, new_width);
    timer.Stop();
    int surplus = new_N % block_size;

    cudaMemcpy(h_new_image, d_new_image, NEW_TOTAL_BYTES, cudaMemcpyDeviceToHost);

    for (int i = new_N - surplus; i < new_N; i+=4) {
        int new_idx = BYTES_PER_PIXEL * i;
        int actual_idx = BYTES_PER_PIXEL * convert_to_actual_idx(new_idx, width, new_width);
        float sum_r, sum_g, sum_b;
        sum_r = 0.0;
        sum_g = 0.0;
        sum_b = 0.0;
        int ii, jj;
        for (ii = 0; ii < MATRICES_SIZE; ii++) {
            for (jj = 0; jj < MATRICES_SIZE; jj++) {
                int idx;
                idx = BYTES_PER_PIXEL * idx_to_ij(actual_idx, ii, jj, width);
                sum_r += h_image[idx] * w[MATRICES_SIZE * ii + jj];
                sum_g += h_image[idx + 1] * w[MATRICES_SIZE * ii + jj];
                sum_b += h_image[idx + 2] * w[MATRICES_SIZE * ii + jj];
            }
        }
        //clamp for R
        sum_r = (sum_r < 0) ? 0 : sum_r;
        sum_r = (sum_r > 255) ? 255 : sum_r;
        //clamp for G
        sum_g = (sum_g < 0) ? 0 : sum_g;
        sum_g = (sum_g > 255) ? 255 : sum_g;
        //clamp for B
        sum_b = (sum_b < 0) ? 0 : sum_b;
        sum_b = (sum_b > 255) ? 255 : sum_b;

        h_new_image[new_idx] = (unsigned char)sum_r;
        h_new_image[new_idx + 1] = (unsigned char)sum_g;
        h_new_image[new_idx + 2] = (unsigned char)sum_b;
        h_new_image[new_idx + 3] = 255;
	}

    lodepng_encode32_file(output_filename, h_new_image, new_width, new_height);
    printf("Convolution Done!\n");
    printf("Time elapsed = %g ms\n", timer.Elapsed());

	cudaFree(d_image);
    cudaFree(d_new_image);

    return 0;
}

