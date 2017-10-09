/******************************************************************************
* PROGRAM: copyStruture
* PURPOSE: This program is a test which test the ability to transfer C  
* 	structured data from host to device, modify them and transfer back.
*
*
* NAME: Vuong Pham-Duy.
	College student.
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

typedef struct {
	int d1, d2, d3;
} data;

__global__
void small_change(data* data)
{
	data->d1 = 2;
	printf("[kernel]: d1 = %d\n", data->d1);
	data->d2 = 3;
	data->d3 = 4;
}

int main(int argc, char *argv[])
{
	data* h_dat;
	data* d_dat;
	h_dat = (data*)malloc(sizeof(data));
	h_dat->d1 = 1;
	h_dat->d2 = 2;
	h_dat->d3 = 3;
	printf("d1 = %d\n", h_dat->d1);
	gpuErrchk(cudaMalloc((void**)&d_dat, sizeof(data)));

	gpuErrchk(cudaMemcpy(d_dat, h_dat, sizeof(data), \
		cudaMemcpyHostToDevice));

	small_change<<<1, 1>>>(d_dat);	 	
	
	gpuErrchk(cudaMemcpy(h_dat, d_dat, sizeof(data), \
		cudaMemcpyDeviceToHost));
	
	printf("d1 = %d\n", h_dat->d1);
	return 0;
}
