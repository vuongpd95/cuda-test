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

/* Structure *****************************************************************/
typedef struct {
	int64_t rbeg;
	int32_t qbeg, len;
	int score;
} mem_seed_t; // unaligned memory

typedef struct {
	int n, m, first, rid;
	uint32_t w:29, kept:2, is_alt:1;
	float frac_rep;
	int64_t pos;
	mem_seed_t *seeds;
} mem_chain_t;

typedef struct {
	size_t n, m;
	mem_chain_t *a;
} mem_chain_v;

typedef struct {
	int n, m, first, rid;
	uint32_t w:29, kept:2, is_alt:1;
	float frac_rep;
	int64_t pos;
} flat_mem_chain_t;

typedef struct {
	size_t n, m;
} flat_mem_chain_v;

__global__ void func(int n, int n_a, int n_seeds, flat_mem_chain_v *chns, \
		flat_mem_chain_t *a, mem_seed_t *seeds, int *d_b) {
	int i, j;
	*d_b = 0;
	for(i = 0; i < n; i++) {
		*d_b += chns[i].n;
		for(j = 0; j < n_a; j++) {
			*d_b += a[j].n;		
		}	
	}
}
/* Structure *****************************************************************/
int main(int argc, char *argv[])
{
	// Assumptions
	int b;
	int *d_b;
	gpuErrchk(cudaMalloc(&d_b, sizeof(int)));
	// Begin Assumptions
	int n, i, j, k;
	n = 10;
	
	mem_chain_v *chns;
	chns = (mem_chain_v*)malloc(sizeof(mem_chain_v) * n);
	for(i = 0; i < n; i++) {
		chns[i].n = 10;
		chns[i].a = (mem_chain_t*)malloc(\
			chns[i].n * sizeof(mem_chain_t));
		for(j = 0; j < chns[i].n; j++) {
			chns[i].a[j].n = 10;
			chns[i].a[j].seeds = (mem_seed_t*)malloc(\
				chns[i].a[j].n * sizeof(mem_seed_t));		
			for(k = 0; k < chns[i].a[j].n; k++) {
				chns[i].a[j].seeds[k].score = i + j + k;
			}
		}
	}
	// End Assumptions
	int n_a, n_seeds;
	n_a = 0; n_seeds = 0;

	for(i = 0; i < n; i++) {
		n_a += chns[i].n;
		for(j = 0; j < chns[i].n; j++) {
			n_seeds += chns[i].a[j].n;
		}
	}

	flat_mem_chain_v *f_chns, *df_chns;
	flat_mem_chain_t *f_a, *df_a;
	mem_seed_t *seeds, *d_seeds;
	
	// Flattened the nested structure
	f_chns = (flat_mem_chain_v*)malloc(n * sizeof(flat_mem_chain_v));
	f_a = (flat_mem_chain_t*)malloc(n_a * sizeof(flat_mem_chain_t));
	seeds = (mem_seed_t*)malloc(n_seeds * sizeof(mem_seed_t));

	int acc_a, acc_seeds;
	acc_a = 0; acc_seeds = 0;

	for(i = 0; i < n; i++) {
		f_chns[i].n = chns[i].n;
		f_chns[i].m = chns[i].m;
		for(j = 0; j < chns[i].n; j++) {
			// int n, m, first, rid;
			// uint32_t w:29, kept:2, is_alt:1;
			// float frac_rep;
			// int64_t pos;
			mem_chain_t *tmp;
			tmp = &chns[i].a[j];
			f_a[acc_a].n = tmp->n;
			f_a[acc_a].m = tmp->m;
			f_a[acc_a].first = tmp->first;
			f_a[acc_a].rid = tmp->rid;
			f_a[acc_a].w = tmp->w;
			f_a[acc_a].kept = tmp->kept;
			f_a[acc_a].is_alt = tmp->is_alt;
			f_a[acc_a].frac_rep = tmp->frac_rep;
			f_a[acc_a].pos = tmp->pos;
			for(k = 0; k < chns[i].a[j].n; k++) {
				// int64_t rbeg;
				// int32_t qbeg, len;
				// int score;
				mem_seed_t *tmp0;
				tmp0 = &chns[i].a[j].seeds[k];
				seeds[acc_seeds].rbeg = tmp0->rbeg;
				seeds[acc_seeds].qbeg = tmp0->qbeg;
				seeds[acc_seeds].len = tmp0->len;
				seeds[acc_seeds].score = tmp0->score;
			}
			acc_seeds += chns[i].a[j].n;		
		}
		acc_a += chns[i].n;
	}

	// Copy the flattened structure to kernel
	gpuErrchk(cudaMalloc(&df_chns, n * sizeof(flat_mem_chain_v)));
	gpuErrchk(cudaMalloc(&df_a, n_a * sizeof(flat_mem_chain_t)));
	gpuErrchk(cudaMalloc(&d_seeds, n_seeds * sizeof(mem_seed_t)));

	gpuErrchk(cudaMemcpy(df_chns, f_chns, n * sizeof(flat_mem_chain_v), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(df_a, f_a, n_a * sizeof(flat_mem_chain_t), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_seeds, seeds, n_seeds * sizeof(mem_seed_t), \
				cudaMemcpyHostToDevice));	

	printf("n = %d, n_a = %d\n", n, n_a);
	func<<<1, 1>>>(n, n_a, n_seeds, df_chns, df_a, d_seeds, d_b);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(&b, d_b, sizeof(int), cudaMemcpyDeviceToHost));	
	printf("d_b = %d\n", b);
	
	cudaFree(df_chns);
	cudaFree(df_a);
	cudaFree(d_seeds);
	cudaFree(d_b);
	return 0;
}
