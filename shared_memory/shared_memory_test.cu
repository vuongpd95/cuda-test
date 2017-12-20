/******************************************************************************
* PROGRAM: copyStruture
* PURPOSE: This program is a test which test the ability to transfer multilevel 
*	C++ structured data from host to device, modify them and transfer back.
*
*
* NAME: Vuong Pham-Duy.
*	College student.
*       Faculty of Computer Science and Technology.
*       Ho Chi Minh University of Technology, Viet Nam.
*       vuongpd95@gmail.com
*
* DATE: 5/10/2017
*
******************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", \
			cudaGetErrorString(code), file, line);
		if (abort) 
			exit(code);
	}
}

__global__ 
void func(int *num) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int s_num[8];
	s_num[threadIdx.x] = num[tid];
	__syncthreads();
	if (tid == 8) {
		for(int i = 0; i < 8; i++) {
			printf("In global: num[%d] = %d | ", tid + i, num[tid + i]);
			printf("In shared: s_num[%d] = %d.\n", threadIdx.x + i, s_num[threadIdx.x + i]);
		}
	}
}

int main(int argc, char *argv[])
{
	int *num = (int*)malloc(8 * 8 * sizeof(int));
	for(int i = 0; i < 8 * 8; i++) num[i] = i;
	int *d_num;
	gpuErrchk(cudaMalloc(&d_num, 8 * 8 * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_num, num, 8 * 8 * sizeof(int), \
		cudaMemcpyHostToDevice));
	dim3 thread_per_block(8);
	int num_block = 8;
	func<<<num_block, thread_per_block>>>(d_num);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	return 0;
}
