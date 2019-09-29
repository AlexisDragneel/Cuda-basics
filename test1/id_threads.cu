#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void print_threadIds() {
	printf("threadIdx.x: %d, threadidx.y: %d, threadIdx.z: %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void print_details() {
	printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, blockDim.x: %d, blockDim.y: %d, blockDim.y: %d, gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n",
		blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

__global__ void print_3d_threads() {
	printf("threadIdx.x: %d, threadidx.y: %d, threadIdx.z: %d, gridDim.x: %d, gridDim.y: %d, gridDim.z: %d \n", 
		threadIdx.x, threadIdx.y, threadIdx.z, gridDim.x, gridDim.y, gridDim.z);

}

__global__ void unique_idx_calc_threadIdx(int* input) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	printf("threadIdx: %d, value: %d \n", tid, input[tid]);
}

__global__ void calculate_unique_gid() {
	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int gid = tid + offset;
	// index = tid + offset
	printf("threadIdx.x: %d, blockIdx.X: %d, blocDim.x: %d, gid: %d  \n",
		tid, blockIdx.x, blockDim.x, gid);
}

__global__ void unique_gid_calculation_2d_in_1d_array_1d_block(int* input ) {
	printf("threadIdx.x: %d, threadIdx.y: %d \n", threadIdx.x, threadIdx.y);
	printf("blockIdx.x: %d, blocIdx.y: %d \n", blockIdx.x, blockIdx.y);
	printf("blockDim.x: %d, blocDim.y: %d \n", blockDim.x, blockDim.y);
	printf("gid1d: %d, \n", threadIdx.x + (blockIdx.x + blockDim.x));

	// index = row offset + block ofsset  + tid;
	// index = number of threads in one row * blockIdx.y + number of threads in thread block * blockIdx.x + threadIdx.x
	// number of threads in one row = gridDim.x * blockDim.x
	int tid = threadIdx.x;
	int rowOffset = blockIdx.y * (gridDim.x * blockDim.x);
	int blockOffset = blockIdx.x * blockDim.x;
	int gid = tid + rowOffset + blockOffset;
	printf("gid: %d \n", gid);
	printf("value: %d \n", input[gid]);
}

__global__ void unique_gid_calculation_2d_in_1d_array_2d_block(int* input) {

	// tid = threadIdx.y * blockDim.x + threadIdx.x
	// block_offset = number of thread in a block (blockDim.x * blockDim.y) * blockIdx.x
	// row_offset = number of Threads in a row (blockDim.x blockDim.y * gridDim.x)  * blockIdx.y;
	// index = tid + block_offset + row_offset
	int tid = threadIdx.x + (blockDim.x * threadIdx.y);
	
	int num_of_threads_in_a_block = blockDim.x * blockDim.y;
	int block_offset = blockIdx.x * num_of_threads_in_a_block;

	int num_of_threads_in_a_row = num_of_threads_in_a_block * gridDim.x;
	int row_offset = num_of_threads_in_a_row * blockIdx.y;
	
	int gid = tid + block_offset + row_offset;
	printf("tid: %d, gid: %d, value: %d \n", tid, gid, input[gid]);
}

__global__ void mem_trs_test (int* input){
	int gid = threadIdx.x + (blockIdx.x + threadIdx.x);
	printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
}

__global__ void mem_trs_test2(int* input, int size) {
	int gid = threadIdx.x + (blockIdx.x + threadIdx.x);
	if (gid < size)
		printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
}

int main() {
	/**int nx, ny, nz;
	nx = 8;
	ny = 8;
	nz = 8;

	dim3 block(2, 2, 2);
	dim3 grid(nx / block.x, ny / block.y, nz/block.z);

	print_3d_threads << < grid, block>>> ();
	cudaDeviceSynchronize();

	cudaDeviceReset(); **/

	/**int array_size = 8;
	int array_byte_size = sizeof(int) * array_size;
	printf("%d \n",array_byte_size);
	int h_data[] = { 23,9,4,53,65,12,1,33 };

	for (int i = 0; i < array_size; i++) {
		printf("%d, ", h_data[i]);
	}

	printf("\n \n");

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice); **/

	/**int array_size = 16;
	int array_byte_size = sizeof(int) * array_size;
	printf("%d \n", array_byte_size);
	int h_data[] = { 23,9,4,53,65,12,1,33, 23,9,4,53,65,12,1,33 };

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(2, 2);
	dim3 grid(2, 2);

	unique_gid_calculation_2d_in_1d_array_2d_block << <grid, block >> > (d_data);**/

	int size = 150;
	int byte_size = size * sizeof(int);

	int* h_input;
	h_input = (int*)malloc(byte_size);

	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		h_input[i] = (int)(rand() & 0xff);
	}

	int* d_input;
	cudaMalloc((void**)&d_input, byte_size);
	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
	dim3 block(32);
	dim3 grid(5);

	mem_trs_test2 << < grid, block >>> (d_input, size);

	cudaDeviceSynchronize();

	cudaDeviceReset();



	return 0;
}