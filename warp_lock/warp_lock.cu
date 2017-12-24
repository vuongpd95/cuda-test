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
#include <inttypes.h>

#define WARP 32
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


__device__ bool warp_lock(int req){
  return ((__ffs(__ballot(req))) == ((threadIdx.x & 31)+1));
}

__global__ void test_lock(){

	bool done;
	int myreq = 1;	
	do {
		done = false;
		__syncthreads();
		// attempt to  "acquire lock"
		bool mylock = warp_lock(myreq);
		// if lock acquired, do "critical section"
		if (mylock){
			done = true;
			printf("Thread %d\n", threadIdx.x);
			myreq = 0;
		}
		__syncthreads();
	} while (!done);
}

int main(){

	test_lock<<<1, WARP>>>();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	return 0;
}
