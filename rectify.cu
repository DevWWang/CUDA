#include <stdio.h>
#include "lodepng.h"
#include "gputimer.h"

#define BYTES_PER_PIXEL 4

__global__ void rectify(unsigned char *d_new_image, unsigned char *d_image){
    int actual_idx = BYTES_PER_PIXEL * (blockDim.x * blockIdx.x + threadIdx.x);
    d_new_image[actual_idx] = d_image[actual_idx] < 127 ? 127 : d_image[actual_idx];
    d_new_image[actual_idx + 1] = d_image[actual_idx + 1] < 127 ? 127 : d_image[actual_idx + 1];
    d_new_image[actual_idx + 2] = d_image[actual_idx + 2] < 127 ? 127 : d_image[actual_idx + 2];
    d_new_image[actual_idx + 3] = d_image[actual_idx + 3];
}

int main(int argc, char **argv) {
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    printf("rectifying %s...\n", input_filename);

    unsigned error;
    unsigned char *h_image, *h_new_image;
    unsigned width, height;

    GpuTimer timer;

    error = lodepng_decode32_file(&h_image, &width, &height, input_filename);
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

    const int N = width * height;
    const int TOTAL_BYTES = N * BYTES_PER_PIXEL * sizeof(unsigned char);

    h_new_image = (unsigned char *) malloc(TOTAL_BYTES);

    unsigned char *d_image;
    unsigned char *d_new_image;

    cudaMalloc(&d_image, TOTAL_BYTES);
    cudaMalloc(&d_new_image, TOTAL_BYTES);

    cudaMemcpy(d_image, h_image, TOTAL_BYTES, cudaMemcpyHostToDevice);

	int block_size = 1024;
    int chunk_size = N / block_size;
    timer.Start();
    rectify<<<chunk_size, block_size>>>(d_new_image, d_image);
    timer.Stop();
	int surplus = N % block_size;

    cudaMemcpy(h_new_image, d_new_image, TOTAL_BYTES, cudaMemcpyDeviceToHost);

	for (int i = N - surplus; i < N; i++) {
		int actual_idx = BYTES_PER_PIXEL * i;
	    h_new_image[actual_idx] = h_image[actual_idx] < 127 ? 127 : h_image[actual_idx];
	    h_new_image[actual_idx + 1] = h_image[actual_idx + 1] < 127 ? 127 : h_image[actual_idx + 1];
	    h_new_image[actual_idx + 2] = h_image[actual_idx + 2] < 127 ? 127 : h_image[actual_idx + 2];
      	h_new_image[actual_idx + 3] = h_image[actual_idx + 3];
	}

    lodepng_encode32_file(output_filename, h_new_image, width, height);
    printf("Rectification Done!\n");
    printf("Time elapsed = %g ms\n", timer.Elapsed());

	cudaFree(d_image);
    cudaFree(d_new_image);

    return 0;
}

