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

__global__ void func(int *value) {
	printf("value[%d] = %d\nvalue[%d] = %d\n", \
		value[0], value[0], value[1], value[1]);
}

int main(int argc, char *argv[])
{
	int *value;
	gpuErrchk(cudaMallocManaged(&value, 2 * sizeof(int)));
	value[0] = 0;
	value[1] = 1;
	func<<<1, 1>>>(value);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaFree(value));
	return 0;
}
