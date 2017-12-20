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
	int64_t rb, re; // [rb,re): reference sequence in the alignment
	int qb, qe;     // [qb,qe): query sequence in the alignment
	int rid;        // reference seq ID
	int score;      // best local SW score
	int truesc;     // actual score corresponding to the aligned region; possibly smaller than $score
	int sub;        // 2nd best SW score
	int alt_sc;
	int csub;       // SW score of a tandem hit
	int sub_n;      // approximate number of suboptimal hits
	int w;          // actual band width used in extension
	int seedcov;    // length of regions coverged by seeds
	int secondary;  // index of the parent hit shadowing the current hit; <0 if primary
	int secondary_all;
	int seedlen0;   // length of the starting seed
	int n_comp:30, is_alt:2; // number of sub-alignments chained together
	float frac_rep;
	uint64_t hash;
} mem_alnreg_t;

typedef struct {
	size_t n, m; 
	mem_alnreg_t *a; 
} mem_alnreg_v;

typedef struct {
	size_t n, m;
} flat_mem_alnreg_v;
/*****************************************************************************/

__global__ 
void func_0(int *n_a, flat_mem_alnreg_v *f_av, mem_alnreg_v **avs) {
	// Assumptions
	int idx;
	idx = blockIdx.x;
	mem_alnreg_v *av;
	int i;	
	av = (mem_alnreg_v*)malloc(sizeof(mem_alnreg_v));
	av->n = 10;
	av->m = 15;
	av->a = (mem_alnreg_t*)malloc(av->n * sizeof(mem_alnreg_t));
	for(i = 0; i < av->n; i++) av->a[i].score = i;
	// End assumptions
	avs[idx] = av;
	atomicAdd(n_a, av->n);
	f_av[idx].n = av->n;
	f_av[idx].m = av->m;
}

__global__ 
void func_1(mem_alnreg_v **avs, int *i_a, mem_alnreg_t *a) {
	int idx, i, size;
	idx = blockIdx.x;
	i = i_a[idx];
	size = avs[idx]->n;
	memcpy(&a[i], avs[idx]->a, size * sizeof(mem_alnreg_t));
}

int main(int argc, char *argv[])
{
	// Begin Assumptions
	int n, i; 
	n = 10;	
	// End Assumptions

	// For confirmation
	mem_alnreg_v **d_avs;
	int h_na, *d_na, *i_a, *di_a;
	flat_mem_alnreg_v *h_fav, *d_fav;
	mem_alnreg_t *h_a, *d_a;

	h_fav = (flat_mem_alnreg_v*)malloc(n * sizeof(flat_mem_alnreg_v));
	gpuErrchk(cudaMalloc(&d_fav, n * sizeof(flat_mem_alnreg_v)));

	gpuErrchk(cudaMalloc(&d_na, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_avs, n * sizeof(mem_alnreg_v*)));
	// End for confirmation
	
	// Copy the flattened structure to kernel
	int num_block;
	dim3 thread_per_block(1);
	num_block = 10;
	
	func_0<<<num_block, thread_per_block>>>(d_na, d_fav, d_avs);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaMemcpy(&h_na, d_na, sizeof(int), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaMemcpy(h_fav, d_fav, n * sizeof(flat_mem_alnreg_v), \
			cudaMemcpyDeviceToHost));
	
	h_a = (mem_alnreg_t*)malloc(h_na * sizeof(mem_alnreg_t));
	i_a = (int*)malloc(n * sizeof(int));
	
	int acc_a;
	acc_a = 0;
	for(i = 0; i < n; i++) {
		i_a[i] = acc_a;
		acc_a += h_fav[i].n;	
	}
	
	gpuErrchk(cudaMalloc(&di_a, n * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_a, h_na * sizeof(mem_alnreg_t)));

	gpuErrchk(cudaMemcpy(di_a, i_a, n * sizeof(int), cudaMemcpyHostToDevice));

	func_1<<<num_block, thread_per_block>>>(d_avs, di_a, d_a);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_a, d_a, h_na * sizeof(mem_alnreg_t), cudaMemcpyDeviceToHost));

	printf("Give me an i: ");
	scanf("%d", &i);
	printf("h_avs[%d].n = %lu, h_avs[%d].m = %lu.\n", i, h_fav[i].n, i, h_fav[i].m);
	int j;
	for(j = 0; j < h_fav[i].n; j++) {
		printf("h_avs[%d].a[%d].score = %d.\n", i, j, h_a[i_a[i] + j].score);	
	}
	/**/
}

