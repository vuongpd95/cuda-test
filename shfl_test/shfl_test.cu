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

__global__ void bcast(int arg) {
	int laneId = threadIdx.x % 32; 
	int value; 
	if (laneId == 0) 			// Note unused variable for 
		value = arg; 			// all threads except lane 0 
	value = __shfl(value, 0, 32); // Synchronize all threads in warp, and get "value" from lane 0 
	if (value != arg) 
		printf("Thread %d failed. value = %d\n", threadIdx.x, value);
	else
		printf("Thread %d success. value = %d\n", threadIdx.x, value); 
} 

int main() { 
	bcast<<< 1, 32 >>>(1234); 
	cudaDeviceSynchronize(); 
	return 0; 
}


