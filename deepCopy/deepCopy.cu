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
/************************** Test 1 lv nested structure ************************/
typedef struct {
	int64_t offset;
	int32_t len;
	int32_t n_ambs;
	uint32_t gi;
	int32_t is_alt;
} bntann1_t;

typedef struct {
	int64_t l_pac;
	int32_t n_seqs;
	bntann1_t *anns; // n_seqs elements
} bntseq_t;

void bns_to_device(const bntseq_t *bns, bntseq_t **d_bns);

__global__ void func0(bntann1_t *anns, int64_t l_pac, int32_t n_seqs, \
	int *d_b) {
	*d_b = n_seqs + anns[0].offset;
}
/************************** Test 1 lv nested structure ************************/
/************************** Test __constant__ *********************************/
typedef struct {
	int b1;
	int b2;
	int b3;
} burge;

__constant__ burge opt;

__global__ void func(int *d_a) {
	*d_a = opt.b1 + opt.b2 + opt.b3;
}
/************************** Test __constant__ *********************************/
int main(int argc, char *argv[])
{
/************************** Test __constant__ *********************************/
	const burge *pb;
	burge bu;
	bu.b1 = 0;
	bu.b2 = 1;
	bu.b3 = 2;
	pb = &bu;

	int a;
	int *d_a;
	a = 1;
	gpuErrchk(cudaMalloc(&d_a, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(opt, pb, sizeof(burge), 0, \
		cudaMemcpyHostToDevice));
	func<<<1, 1>>>(d_a);
	gpuErrchk(cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost));

	printf("a = %d\n", a);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
/************************** Test __constant__ *********************************/	
/************************** Test 1 lv nested structure ************************/
	// Assumptions
	bntseq_t *bns;
	bns = (bntseq_t*)malloc(sizeof(bntseq_t));
	bns->l_pac = 1;
	bns->n_seqs = 2500;
	bns->anns = (bntann1_t*)malloc(bns->n_seqs * sizeof(bntann1_t));
	bns->anns[0].offset = 10;	
	const bntseq_t *cbns = bns;

	int b;
	int *d_b;
	gpuErrchk(cudaMalloc(&d_b, sizeof(int)));
	//
	bntann1_t *d_anns;

	gpuErrchk(cudaMalloc(&d_anns, cbns->n_seqs * sizeof(bntann1_t)));
	gpuErrchk(cudaMemcpy(d_anns, cbns->anns, \
		cbns->n_seqs * sizeof(bntann1_t), cudaMemcpyHostToDevice));

	func0<<<1, 1>>>(d_anns, cbns->l_pac, cbns->n_seqs, d_b);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(&b, d_b, sizeof(int), cudaMemcpyDeviceToHost));
	
	printf("b = %d\n", b);
	printf("b = %ld\n", bns->n_seqs + bns->anns[0].offset);
/************************** Test 1 lv nested structure ************************/
	return 0;
}

