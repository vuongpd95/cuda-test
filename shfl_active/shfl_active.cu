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
#define LANE 0
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
void func_0(int qlen, eh_t *d_qp, int tlen, int max_pass, int t_lastp) {
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int lane_id = threadIdx.x % WARP;
	int in_h, in_e;
	int out_h, out_e;
	int num_active, beg;

	for(int i = 0; i < max_pass; i++) {
		beg = 0;
		reset(&in_h, &in_e); reset(&out_h, &out_e);
		if(lane_id == 0) {
			out_h = d_qp[beg].h;
			out_e = d_qp[beg].e;
			beg += 1;
		}
		__syncthreads();
		if(i == max_pass - 1) {
			if(lane_id >= t_lastp) break;
			else num_active = t_lastp;
		} else num_active = WARP;

		if(lane_id == LANE) printf("Lane %d, received: ", lane_id);
		// We can not use shfl when two threads are in different branch (when use if/else, while/do)
		do {
			// Keep shfl cmds outside of all if/else
			in_h = __shfl(out_h, lane_id - 1, 32);
			in_e = __shfl(out_e, lane_id - 1, 32);

			if(lane_id == 0) {
				in_h = d_qp[beg].h;
				in_e = d_qp[beg].e;
			}
			//__syncthreads();
			if(check_active(in_h, in_e)) {
				if(lane_id == LANE) printf("[%d, %d] ", in_h, in_e);
				if(lane_id != num_active - 1) {
					out_h = in_h;
					out_e = in_e;
				} else {
					if(i != max_pass - 1) {
						d_qp[beg].h = in_h + 1;
						d_qp[beg].e = in_e + 1;
					}
				}

				reset(&in_h, &in_e);
				beg += 1;
			} else {
				if(lane_id == LANE) printf("[nothing] ");
			}
		} while((beg < qlen + 1) || (lane_id != (num_active - 1) && beg < qlen + 2));
		// The above condition keeps all threads other than the last one inside the loop which enable the
		// (lane_id + 1) thread to use shfl
		if(lane_id == LANE) printf("\n");
	}
}

int main(int argc, char *argv[])
{
	int qlen, tlen;
	printf("Input qlen = ");
	scanf("%d", &qlen);

	printf("Input tlen = ");
	scanf("%d", &tlen);

	eh_t *h_qp, *d_qp;
	h_qp = (eh_t*)malloc(sizeof(eh_t) * (qlen + 1));
	for(int i = 0; i <= qlen; i++) {
		h_qp[i].h = i + 10;
		h_qp[i].e = i + 10;
	}
	gpuErrchk(cudaMalloc(&d_qp, sizeof(eh_t) * (qlen + 1)));
	gpuErrchk(cudaMemcpy(d_qp, h_qp, sizeof(eh_t) * (qlen + 1), cudaMemcpyHostToDevice));


	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	int max_pass = (int)((float)tlen/(float)WARP + 1.);
	int t_lastp = tlen - (tlen/WARP)*WARP;

	func_0<<<1, WARP>>>(qlen, d_qp, tlen, max_pass, t_lastp);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	fprintf(stderr, "[M::%s] Kernel executed in %f ms\n" , __func__, elapsedTime);

	return 0;
	
}

