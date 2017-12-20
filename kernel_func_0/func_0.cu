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
#include <stdint.h>

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
/***************************** Structure *************************************/

typedef struct {
	int l_seq;
	uint8_t *seq;
} bseq2_t;
/*****************************************************************************/
__device__
bseq2_t* get_bseq2_t(int n, int *l_seq, uint8_t *seq, int *i_seq, int i) {
	bseq2_t *ret;
	if(i < 0 || i >= n) 
		return get_bseq2_t(n, l_seq, seq, i_seq, \
					(i >= n) ? ((n - 1) : 0);
	ret = (bseq2_t*)malloc(sizeof(bseq2_t));
	ret->l_seq = l_seq[i];
	ret->seq = (uint8_t*)malloc(ret->l_seq * sizeof(uint8_t));
	memcpy(ret->seq, &seq[i_seq[i]], ret->l_seq * sizeof(uint8_t));
	return ret;
}

__device__ 
void free_bseq2_t(bseq2_t **p) {
	free((*p)->seq);
	free(*p);
}
__global__ 
void func(int n, int *ret_l, uint8_t *ret_seq, int *l_seq, int *i_seq, \
		uint8_t *seq, int i) {
	bseq2_t *ret;
	ret = get_bseq2_t(n, l_seq, seq, i_seq, i);
	memcpy(ret_l, &ret->l_seq, sizeof(int));
	memcpy(ret_seq, ret->seq, ret->l_seq * sizeof(uint8_t));
}

int main(int argc, char *argv[])
{
	/* Making assumptions */
	int i, j, n, acc_seq;	
	int *l_seq, *d_l_seq, *i_seq, *d_i_seq;
	uint8_t *seq, *d_seq;
	n = 9;
	acc_seq = 0;
	l_seq = (int*)malloc(n * sizeof(int));
	i_seq = (int*)malloc(n * sizeof(int));
	seq = (uint8_t*)malloc(45 * sizeof(uint8_t));
	for(i = 0; i < n; i++) {
		l_seq[i] = i + 1;
		for(j = 0; j < l_seq[i]; j++) seq[acc_seq + j] = l_seq[i];
		i_seq[i] = acc_seq;
		acc_seq += l_seq[i];
	}

	/* Finished making assumptions */
	int ret_l_seq;
	uint8_t ret_seq[10];

	int *d_ret_l_seq;
	uint8_t *d_ret_seq;

	gpuErrchk(cudaMalloc(&d_ret_l_seq, sizeof(int)));
	gpuErrchk(cudaMalloc(&d_ret_seq, 10 * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc(&d_l_seq, n * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_i_seq, n * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_seq, 45 * sizeof(uint8_t)));

	gpuErrchk(cudaMemcpy(d_l_seq, l_seq, n * sizeof(int), \
			cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_i_seq, i_seq, n * sizeof(int), \
			cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_seq, seq, 45 * sizeof(uint8_t), \
			cudaMemcpyHostToDevice));
	
	printf("Enter i = ");
	scanf("%d", &i);

	func<<<1, 1>>>(n, d_ret_l_seq, d_ret_seq, d_l_seq, d_i_seq, d_seq, i);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(&ret_l_seq, d_ret_l_seq, sizeof(int), \
			cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&ret_seq, d_ret_seq, 10 * sizeof(uint8_t), \
			cudaMemcpyDeviceToHost));

	printf("h.l_seq = %d\n", ret_l_seq);
	printf("h.seq = ");
	for(i = 0; i < ret_l_seq; i++) {
		printf("%u ", ret_seq[i]);
	}
	printf("\n");

	cudaFree(d_ret_l_seq);
	cudaFree(d_ret_seq);
	cudaFree(d_l_seq);
	cudaFree(d_i_seq);
	cudaFree(d_seq);
	free(l_seq);
	free(i_seq);
	free(seq);	
}

