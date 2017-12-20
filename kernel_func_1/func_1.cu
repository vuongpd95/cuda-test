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

/*****************************************************************************/
__device__ 
mem_chain_v* get_mem_chain_v(int n, flat_mem_chain_v *f_chns, \
		flat_mem_chain_t *f_a, int *i_a, mem_seed_t *seeds, \
		int *i_seeds, int i) {
	if(i < 0 || i >= n) 
		return get_mem_chain_v(n, f_chns, f_a, i_a, seeds, i_seeds, \
					(i >= n) ? (n - 1) : 0);	
	mem_chain_v *ret;
	int j, k, first_a, first_seed, acc_seeds;
	first_a = i_a[i];
	first_seed = i_seeds[i];
	acc_seeds = 0;

	ret = (mem_chain_v*)malloc(sizeof(mem_chain_v));
	ret->n = f_chns[i].n;
	ret->m = f_chns[i].m;
	ret->a = (mem_chain_t*)malloc(ret->n * sizeof(mem_chain_t));
	for(j = 0; j < ret->n; j++) {
		ret->a[j].n = f_a[first_a + j].n;
		ret->a[j].m = f_a[first_a + j].m;
		ret->a[j].first = f_a[first_a + j].first;
		ret->a[j].rid = f_a[first_a + j].rid;
		ret->a[j].w = f_a[first_a + j].w;
		ret->a[j].kept = f_a[first_a + j].kept;
		ret->a[j].is_alt = f_a[first_a + j].is_alt;
		ret->a[j].frac_rep = f_a[first_a + j].frac_rep;
		ret->a[j].pos = f_a[first_a + j].pos;
		ret->a[j].seeds = (mem_seed_t*)malloc(ret->a[j].n * sizeof(mem_seed_t));
		/*for(k = 0; k < ret->a[j].n; k++) {
			ret->a[j].seeds[k].rbeg = seeds[first_seed + acc_seeds + k].rbeg;
			ret->a[j].seeds[k].qbeg = seeds[first_seed + acc_seeds + k].qbeg;
			ret->a[j].seeds[k].len = seeds[first_seed + acc_seeds + k].len;
			ret->a[j].seeds[k].score = seeds[first_seed + acc_seeds + k].score;
		}*/
		memcpy(ret->a[j].seeds, &seeds[first_seed + acc_seeds], \
			ret->a[j].n * sizeof(mem_seed_t));
		acc_seeds += ret->a[j].n;
	}
	return ret;
}

__device__
void free_mem_chain_v(mem_chain_v **p) {
	int i;
	for(i = 0; i < (*p)->n; i++) {
		free((*p)->a[i].seeds);	
	}
	free((*p)->a);
	free(*p);
}
__global__ 
void func(int n, flat_mem_chain_v *f_chns, flat_mem_chain_t *f_a, \
		int *i_a, mem_seed_t *seeds, int *i_seeds, int i, int *score) {
	mem_chain_v *ret;
	ret = get_mem_chain_v(n, f_chns, f_a, i_a, seeds, i_seeds, i);
	*score = ret->a[i].seeds[i].score;
}

int main(int argc, char *argv[])
{
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
	// For confirmation
	int *h_score, *d_score;
	h_score = (int*)malloc(sizeof(int));
	gpuErrchk(cudaMalloc(&d_score, sizeof(int)));
	// End for confirmation
	int n_a, n_seeds;
	n_a = 0; n_seeds = 0;
	int *i_a, *i_seeds, *di_a, *di_seeds;

	i_a = (int*)malloc(n * sizeof(int));
	i_seeds = (int*)malloc(n * sizeof(int));

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
		i_seeds[i] = acc_seeds;
		f_chns[i].n = chns[i].n;
		f_chns[i].m = chns[i].m;
		for(j = 0; j < chns[i].n; j++) {
			// int n, m, first, rid;
			// uint32_t w:29, kept:2, is_alt:1;
			// float frac_rep;
			// int64_t pos;
			mem_chain_t *tmp;
			tmp = &chns[i].a[j];
			f_a[acc_a + j].n = tmp->n;
			f_a[acc_a + j].m = tmp->m;
			f_a[acc_a + j].first = tmp->first;
			f_a[acc_a + j].rid = tmp->rid;
			f_a[acc_a + j].w = tmp->w;
			f_a[acc_a + j].kept = tmp->kept;
			f_a[acc_a + j].is_alt = tmp->is_alt;
			f_a[acc_a + j].frac_rep = tmp->frac_rep;
			f_a[acc_a + j].pos = tmp->pos;
			for(k = 0; k < chns[i].a[j].n; k++) {
				// int64_t rbeg;
				// int32_t qbeg, len;
				// int score;
				mem_seed_t *tmp0;
				tmp0 = &chns[i].a[j].seeds[k];
				seeds[acc_seeds + k].rbeg = tmp0->rbeg;
				seeds[acc_seeds + k].qbeg = tmp0->qbeg;
				seeds[acc_seeds + k].len = tmp0->len;
				seeds[acc_seeds + k].score = tmp0->score;
			}
			acc_seeds += chns[i].a[j].n;		
		}
		i_a[i] = acc_a;
		acc_a += chns[i].n;
	}

	// Copy the flattened structure to kernel
	gpuErrchk(cudaMalloc(&df_chns, n * sizeof(flat_mem_chain_v)));
	gpuErrchk(cudaMalloc(&df_a, n_a * sizeof(flat_mem_chain_t)));
	gpuErrchk(cudaMalloc(&d_seeds, n_seeds * sizeof(mem_seed_t)));
	gpuErrchk(cudaMalloc(&di_a, n * sizeof(int)));
	gpuErrchk(cudaMalloc(&di_seeds, n * sizeof(int)));

	gpuErrchk(cudaMemcpy(df_chns, f_chns, n * sizeof(flat_mem_chain_v), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(df_a, f_a, n_a * sizeof(flat_mem_chain_t), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_seeds, seeds, n_seeds * sizeof(mem_seed_t), \
				cudaMemcpyHostToDevice));	
	gpuErrchk(cudaMemcpy(di_a, i_a, n * sizeof(int), \
				cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(di_seeds, i_seeds, n * sizeof(int), \
				cudaMemcpyHostToDevice));

	printf("n = %d, n_a = %d\n", n, n_a);
	printf("Give me an i: ");
	scanf("%d", &i);
	printf("Test value chns[%d].a[%d].seeds[%d].score = %d\n", \
		i, i, i, seeds[i_seeds[i]].score);

	printf("Check i_a = ");
	for(j = 0; j < n; j++) printf("%d ", i_a[j]);
	printf("\nCheck i_seeds = ");
	for(j = 0; j < n; j++) printf("%d ", i_seeds[j]);
	printf("\n");
	
	func<<<1, 1>>>(n, df_chns, df_a, di_a, d_seeds, di_seeds, i, d_score);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_score, d_score, sizeof(int), \
				cudaMemcpyDeviceToHost));
	
	printf("h_score = %d\n", *h_score);
	
	cudaFree(df_chns);
	cudaFree(df_a);
	cudaFree(di_a);
	cudaFree(d_seeds);
	cudaFree(di_seeds);
	
	for(i = 0; i < n; i++) {
		for(j = 0; j < chns[i].n; j++) {
			free(chns[i].a[j].seeds);
		}
		free(chns[i].a);
	}
	
	free(chns);
	free(f_chns);
	free(f_a);
	free(i_a);
	free(seeds);
	free(i_seeds);
	// Free confirmation
	cudaFree(d_score);
	free(h_score);		
}

