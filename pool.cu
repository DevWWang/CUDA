#include <stdio.h>
#include <math.h>
#include "lodepng.h"
#include "gputimer.h"

#define BYTES_PER_PIXEL 4
#define BLOCK_SIZE 2
#define max(x, y)  ((x) > (y) ? (x) : (y))

__host__ __device__ int idx_to_ij(int idx, int ii, int jj, unsigned width) {
    int i, j, position;
    i = (idx / BYTES_PER_PIXEL) / width;
    j = (idx / BYTES_PER_PIXEL) % width;
    position = BLOCK_SIZE * ii + jj;

    switch(position){
        case 0:
            return (width * i + j);
        case 1:
            return (width * i + (j + 1));
        case 2:
            return (width * (i + 1) + j);
        case 3:
            return (width * (i + 1) + (j + 1));
        default:
            return 0;
    }
}

__host__ __device__ int convert_to_actual_idx(int idx, unsigned width, unsigned new_width) {
    int i, j;
    i = (idx / BYTES_PER_PIXEL) / new_width;
    j = (idx / BYTES_PER_PIXEL) % new_width;

    return(width * (i * 2) + (j * 2));
}

__global__ void pool(unsigned char *d_new_image, unsigned char *d_image, unsigned width, unsigned new_width){
    int new_idx = BYTES_PER_PIXEL * (blockDim.x * blockIdx.x + threadIdx.x);
    int actual_idx = BYTES_PER_PIXEL * convert_to_actual_idx(new_idx, width, new_width);

    int idx_1 = BYTES_PER_PIXEL * idx_to_ij(actual_idx, 0, 0, width);
    int idx_2 = BYTES_PER_PIXEL * idx_to_ij(actual_idx, 0, 1, width);
    int idx_3 = BYTES_PER_PIXEL * idx_to_ij(actual_idx, 1, 0, width);
    int idx_4 = BYTES_PER_PIXEL * idx_to_ij(actual_idx, 1, 1, width);

    d_new_image[new_idx] = (unsigned char)max(max(d_image[idx_1], d_image[idx_2]),max(d_image[idx_3], d_image[idx_4]));
    d_new_image[new_idx + 1] = (unsigned char)max(max(d_image[idx_1 + 1], d_image[idx_2 + 1]),max(d_image[idx_3 + 1], d_image[idx_4 + 1]));
    d_new_image[new_idx + 2] = (unsigned char)max(max(d_image[idx_1 + 2], d_image[idx_2 + 2]),max(d_image[idx_3 + 2], d_image[idx_4 + 2]));
    d_new_image[new_idx + 3] = (unsigned char)max(max(d_image[idx_1 + 3], d_image[idx_2 + 3]),max(d_image[idx_3 + 3], d_image[idx_4 + 3]));
}

int main(int argc, char **argv) {
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    printf("pooling %s...\n", input_filename);

    unsigned error;
    unsigned char *h_image, *h_new_image;
    unsigned width, height;

    GpuTimer timer;

    error = lodepng_decode32_file(&h_image, &width, &height, input_filename);
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

    const int N = width * height;
    const int TOTAL_BYTES = N * BYTES_PER_PIXEL * sizeof(unsigned char);

    unsigned new_width = width / 2;
    unsigned new_height = height / 2;
    const int new_N = new_width * new_height;
    const int NEW_TOTAL_BYTES = new_N * BYTES_PER_PIXEL * sizeof(unsigned char);

    h_new_image = (unsigned char *) malloc(NEW_TOTAL_BYTES);

    unsigned char *d_image;
    unsigned char *d_new_image;

    cudaMalloc(&d_image, TOTAL_BYTES);
    cudaMalloc(&d_new_image, NEW_TOTAL_BYTES);

    cudaMemcpy(d_image, h_image, TOTAL_BYTES, cudaMemcpyHostToDevice);

    int block_size = 1024;
    int chunk_size = new_N / block_size;
    timer.Start();
    pool<<<chunk_size, block_size>>>(d_new_image, d_image, width, new_width);
    timer.Stop();
    int surplus = new_N % block_size;

    for (int i = new_N - surplus; i < new_N; i+=4) {
        int new_idx = BYTES_PER_PIXEL * i;
        int actual_idx = BYTES_PER_PIXEL * convert_to_actual_idx(new_idx, width, new_width);

        int idx_1 = BYTES_PER_PIXEL * idx_to_ij(actual_idx, 0, 0, width);
        int idx_2 = BYTES_PER_PIXEL * idx_to_ij(actual_idx, 0, 1, width);
        int idx_3 = BYTES_PER_PIXEL * idx_to_ij(actual_idx, 1, 0, width);
        int idx_4 = BYTES_PER_PIXEL * idx_to_ij(actual_idx, 1, 1, width);

        h_new_image[new_idx] = (unsigned char)max(max(h_image[idx_1], h_image[idx_2]),max(h_image[idx_3], h_image[idx_4]));
        h_new_image[new_idx + 1] = (unsigned char)max(max(h_image[idx_1 + 1], h_image[idx_2 + 1]),max(h_image[idx_3 + 1], h_image[idx_4 + 1]));
        h_new_image[new_idx + 2] = (unsigned char)max(max(h_image[idx_1 + 2], h_image[idx_2 + 2]),max(h_image[idx_3 + 2], h_image[idx_4 + 2]));
        h_new_image[new_idx + 3] = (unsigned char)max(max(h_image[idx_1 + 3], h_image[idx_2 + 3]),max(h_image[idx_3 + 3], h_image[idx_4 + 3]));
    }

    cudaMemcpy(h_new_image, d_new_image, NEW_TOTAL_BYTES, cudaMemcpyDeviceToHost);

    lodepng_encode32_file(output_filename, h_new_image, new_width, new_height);
    printf("Pooling Done!\n");
    printf("Time elapsed = %g ms\n", timer.Elapsed());

    cudaFree(d_image);
    cudaFree(d_new_image);

    return 0;
}

