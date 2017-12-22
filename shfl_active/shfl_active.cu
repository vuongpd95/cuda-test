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
#define LANE 3
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
/* Data structures */
typedef struct {
	int32_t h, e;
} eh_t;

__device__
bool check_active(int32_t h, int32_t e) {
	if(h != -1 && e != -1) return true;
	else return false;
}
__device__
void reset(int32_t *h, int32_t *e) {
	*h = -1;
	*e = -1;
}

__global__ 
void func_0(int qlen, eh_t *d_qp) {
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int lane_id = threadIdx.x % WARP;
	int in_h, in_e;
	int out_h, out_e;
	reset(&in_h, &in_e); reset(&out_h, &out_e);
	int beg = 0;
	if(lane_id == LANE) printf("Lane %d, received: ", lane_id);
	if(lane_id == 0) {
		out_h = d_qp[beg].h;
		out_e = d_qp[beg].e;
		beg += 1;
	}
	__syncthreads();
	
	do {
		if(lane_id == 0) {
			in_h = d_qp[beg].h;
			in_e = d_qp[beg].e;
			
		}
		else {
			in_h = __shfl(out_h, lane_id - 1, WARP);
			in_e = __shfl(out_e, lane_id - 1, WARP);
		}
		__syncthreads();
		if(check_active(in_h, in_e)) {
			if(lane_id == LANE) printf("[%d, %d] ", in_h, in_e);
			out_h = in_h;
			out_e = in_e;
			reset(&in_h, &in_e);
			beg += 1;
		} else {
			if(lane_id == LANE) printf("[nothing] ");
		}
		__syncthreads();	
	} while(beg < qlen);
	if(lane_id == LANE) printf("\n");	
}

int main(int argc, char *argv[])
{
	int qlen;
	printf("Input qlen = ");
	scanf("%d", &qlen);

	eh_t *h_qp, *d_qp;
	h_qp = (eh_t*)malloc(sizeof(eh_t) * (qlen + 1));
	for(int i = 0; i <= qlen; i++) {
		h_qp[i].h = i;
		h_qp[i].e = i;
	}
	gpuErrchk(cudaMalloc(&d_qp, sizeof(eh_t) * (qlen + 1)));
	gpuErrchk(cudaMemcpy(d_qp, h_qp, sizeof(eh_t) * (qlen + 1), cudaMemcpyHostToDevice));


	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	func_0<<<1, WARP>>>(qlen, d_qp);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	fprintf(stderr, "[M::%s] Kernel executed in %f ms\n" , __func__, elapsedTime);

	return 0;
	
}
