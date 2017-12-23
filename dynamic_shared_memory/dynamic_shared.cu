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
#define WARP 32
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

extern __shared__ int32_t container[];
__global__ 
void func(int qlen, int m, int32_t *e, int32_t *h, int8_t *qp) {
	int lane_id = threadIdx.x % WARP;
	int32_t *se, *sh;
	int8_t *sqp;
	sh = container;
	se = (int32_t*)&sh[qlen + 1];
	sqp = (int8_t*)&se[qlen + 1];
	int i = lane_id;
	for(;;) {
		if(i < qlen + 1) {
			sh[i] = h[i];
			se[i] = e[i];		
		}
		if(i < qlen * m) sqp[i] = qp[i];
		else break;
		i += WARP;
	}
	__syncthreads();
	if(lane_id == 0) {
		printf("[h, e]: ");
		for(i = 0; i < qlen + 1; i++) {
			printf("[%d, %d] ", sh[i], se[i]);		
		}	
		printf("\n[qp]: \n");
		for(i = 0; i < qlen * m; i++) {
			printf("%d ", sqp[i]);
			if(i % qlen == 0) printf("\n");		
		}
		printf("\n");
	}
}

int main(int argc, char *argv[])
{
	int qlen, m, i;
	int32_t *e, *h, *d_e, *d_h;
	int8_t *qp, *d_qp;
	printf("Input qlen = ");
	scanf("%d", &qlen);

	printf("Input m = ");
	scanf("%d", &m);

	e = (int32_t*)calloc(qlen + 1, sizeof(int32_t));
	h = (int32_t*)calloc(qlen + 1, sizeof(int32_t));
	qp = (int8_t*)malloc(qlen * m);

	for(i = 0; i < qlen + 1; i++) {
		e[i] = i; h[i] = i; 	
	}
	for(i = 0; i < qlen * m; i++) {
		qp[i] = i % 128;
	}

	gpuErrchk(cudaMalloc(&d_e, sizeof(int32_t) * (qlen + 1)));
	gpuErrchk(cudaMalloc(&d_h, sizeof(int32_t) * (qlen + 1)));
	gpuErrchk(cudaMalloc(&d_qp, sizeof(int8_t) * qlen * m));

	gpuErrchk(cudaMemcpy(d_e, e, sizeof(int32_t) * (qlen + 1), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_h, h, sizeof(int32_t) * (qlen + 1), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_qp, qp, sizeof(int8_t) * qlen * m, cudaMemcpyHostToDevice));	

	func<<<1, WARP, 2 * (qlen + 1) * sizeof(int32_t) + qlen * m * sizeof(int8_t)>>>(qlen, m, d_e, d_h, d_qp);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	return 0;
	
}
