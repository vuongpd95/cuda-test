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
#define LANE 2
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
bool check_active(eh_t data) {
	if(data.h != -1 && data.e != -1) return true;
	else return false;
}
__device__
void reset(eh_t *data) {
	data->h = -1;
	data->e = -1;
}
__global__ 
void func_0(int qlen, eh_t *d_qp) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int lane_id = threadIdx.x % 32;
	eh_t data_out, data_in;
	reset(&data_in); reset(&data_out);
	int beg = 0;
	if(lane_id == LANE) printf("Lane %d, received: ");
	if(lane_id == 0) memcpy(s_qp, d_qp, sizeof(int8_t) * (qlen + 1));
	if(lane_id == 0) {
		data_out = s_qp[beg];
		beg += 1;
	}
	__syncthreads();
	do {
		if(lane_id == 0) data_in = s_qp[beg];
		else data_in = __shfl(data_out, lane_id - 1, WARP);
		if(check_active(data_in)) {
			if(lane_id == LANE) printf("[%ld, %ld] ", data_in.h, data_in.e);
			data_out = data_in;
			reset(&data_in);
			beg += 1;
		} else {
			if(lane_id == LANE) printf("[nothing] ");
		}
	} while(beg < qlen);
}

int main(int argc, char *argv[])
{
	int qlen;
	printf("Input qlen = ");
	scanf("%d", &qlen);

	int8_t *h_qp, *d_qp;
	h_qp = (int8_t*)malloc(sizeof(eh_t) * (qlen + 1)));
	int k = 0;
	for(int i = 0; i <= qlen; i++) {
		h_qp[i].h = h_qp[i].e = i;
	}

	gpuErrchk(cudaMalloc(&d_qp, sizeof(eh_t) * (qlen + 1)));
	gpuErrchk(cudaMemcpy(d_qp, h_qp, sizeof(eh_t) * (qlen + 1), cudaMemcpyHostToDevice));


	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	func_0<<<1, WARP, qlen + 1>>>(qlen, d_qp);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	fprintf(stderr, "[M::%s] Kernel executed in %f ms\n" , __func__, elapsedTime);

	return 0;
	
}
