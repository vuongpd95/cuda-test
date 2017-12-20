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
/* Test Structures */
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

void init_seeds(mem_seed_t *seeds, int len) {
	for(int i = 0; i < len; i++) {
		seeds[i].score = i;
	}
}

void init_chains(mem_chain_t *chains, int len) {
	for(int i = 0; i < len; i++) {
		chains[i].n = i;
	}
}
extern __shared__ float container[];
__device__ unsigned int lock = 0;
__global__ 
void func(mem_seed_t *seeds, int len_seeds, mem_chain_t *chains, int len_chains, int len_container) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	bool leave_loop = false;
	mem_chain_t *shared_chains = (mem_chain_t*)&container;
	mem_seed_t *shared_seeds = (mem_seed_t*)&shared_chains[len_chains];
	while (!leave_loop && threadIdx.x == 0) {
		if (atomicExch(&lock, 1) == 0) {
			// critical section
			memcpy(shared_seeds, seeds, len_seeds * sizeof(mem_seed_t));
			memcpy(shared_chains, chains, len_chains * sizeof(mem_chain_t));
			// end critical section
			leave_loop = true;
			atomicExch(&lock, 0);
		}
	}
	__syncthreads();
	if (tid == 1) {
		for(int i = 0; i < len_seeds; i++) {
			printf("shared_seeds[%d].score = %d\n", i, shared_seeds[i].score);		
		}
		for(int i = 0; i < len_chains; i++) {
			printf("shared_chains[%d].n = %d\n", i, shared_chains[i].n);
		}
	}
}

int main(int argc, char *argv[])
{
	int len_seeds, len_chains;
	len_seeds = 10;
	len_chains = 20;
	
	mem_seed_t *seeds = (mem_seed_t*)malloc(len_seeds * sizeof(mem_seed_t));
	mem_chain_t *chains = (mem_chain_t*)malloc(len_chains * sizeof(mem_chain_t));

	init_seeds(seeds, len_seeds); init_chains(chains, len_chains);
	printf("sizeof(mem_seed_t) = %lu, sizeof(mem_chain_t) = %lu\n", sizeof(mem_seed_t), sizeof(mem_chain_t));
	int len_container = (len_seeds * sizeof(mem_seed_t) + len_chains * sizeof(mem_chain_t)) / sizeof(float);
	
	mem_seed_t *d_seeds;
	mem_chain_t *d_chains;

	gpuErrchk(cudaMalloc(&d_seeds, len_seeds * sizeof(mem_seed_t)));
	gpuErrchk(cudaMalloc(&d_chains, len_chains * sizeof(mem_chain_t)));

	gpuErrchk(cudaMemcpy(d_seeds, seeds, len_seeds * sizeof(mem_seed_t), \
			cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_chains, chains, len_chains * sizeof(mem_chain_t), \
			cudaMemcpyHostToDevice));

	func<<<2, 5, len_container>>>(d_seeds, len_seeds, d_chains, len_chains, len_container);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	return 0;
	
}
