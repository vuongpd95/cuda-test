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

__device__ unsigned int lock[5] = {0, 0, 0, 0, 0};
extern __shared__ int8_t s_qp[];
__global__ 
void func_0(int qlen, int8_t *d_qp) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == 0) memcpy(s_qp, d_qp, sizeof(int8_t) * 5 * qlen);	
	int value = 0;
	bool leave_loop = false;
	while (!leave_loop) {
		if (atomicExch(&lock[tid % 5], 1) == 0) {
			// critical section
			// If all threads only read, then there isn't a need for a locking mechanism
			value = d_qp[(tid % 5) * qlen];
			// end critical section
			leave_loop = true;
			atomicExch(&lock[tid % 5], 0);
		}
	}
	printf("Thread %d, get d_qp[%d] = %d\n", tid, (tid % 5) * qlen, d_qp[(tid % 5) * qlen]);
}

int main(int argc, char *argv[])
{
	int qlen;
	printf("Input qlen = ");
	scanf("%d", &qlen);

	int8_t *h_qp, *d_qp;
	h_qp = (int8_t*)malloc(sizeof(int8_t) * qlen * 5);
	int k = 0;
	for(int i = 0; i < 5; i++) {
		for(int j = 0; j < qlen; j++, k++) h_qp[i * qlen + j] = k;
	}

	gpuErrchk(cudaMalloc(&d_qp, sizeof(int8_t) * qlen * 5));
	gpuErrchk(cudaMemcpy(d_qp, h_qp, sizeof(int8_t) * qlen * 5, cudaMemcpyHostToDevice));


	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	func_0<<<1, 10, 5 * qlen>>>(qlen, d_qp);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	fprintf(stderr, "[M::%s] Kernel executed in %f ms\n" , __func__, elapsedTime);

	return 0;
	
}
