/*
 ============================================================================
 Name        : cuda_lock.cu
 Author      : vuongp
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA thread wide lock, this code works well at the moment but
	there is no guarantee that it will work with all GPU architecture.
 ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

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

__device__ int mLock = 0;

__global__ void func(unsigned int *comm) {
	bool blocked = true;	
	while(blocked) {
	    if(0 == atomicCAS(&mLock, 0, 1)) {
		printf("Block Id = %d, Thread Id = %d acquired lock\n", blockIdx.x, threadIdx.x);
	    	*comm += 1;
	    	printf("Block Id = %d, Thread Id = %d, comm = %u\n", blockIdx.x, threadIdx.x, *comm);
	        atomicExch(&mLock, 0);
		printf("Block Id = %d, Thread Id = %d released lock\n", blockIdx.x, threadIdx.x);
	        blocked = false;
	    }
	}
}
int main(void)
{
	unsigned int *d_comm;
	gpuErrchk(cudaMalloc(&d_comm, sizeof(unsigned int)));
	gpuErrchk(cudaMemset(d_comm, 0, sizeof(unsigned int)));
	func<<<10, 64>>>(d_comm);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	return 0;
}


