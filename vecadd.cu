#include <cuda.h>
#include <stdio.h>

#define N 256

// kernel code for adding two vector elements
__global__ void vecAdd(float* a, float* b, float* c)
{
	int i = threadIdx.x;
	if (i < N)
		c[i] = a[i] + b[i];
}

int main(void)
{
	int   i;
	float a[N], b[N], c[N];
	float *devPtrA, *devPtrB, *devPtrC;

	// initialize arrays
	for (i=0; i < N; i++) {
		a[i] = -i;
		b[i] = i*i;
	}

	// allocate CUDA memory for arrays
	int memsize = N*sizeof(float);
	cudaMalloc((void**)&devPtrA, memsize);
	cudaMalloc((void**)&devPtrB, memsize);
	cudaMalloc((void**)&devPtrC, memsize);

	// copy host data to CUDA memory
	cudaMemcpy(devPtrA, a, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(devPtrB, b, memsize, cudaMemcpyHostToDevice);

	// call add function on CUDA GPU
	vecAdd<<<1, N>>>(devPtrA, devPtrB, devPtrC);

	// copy results back
	cudaMemcpy(c, devPtrC, memsize, cudaMemcpyDeviceToHost);

	// print results
	for (i=0; i < N; i++)
		printf("C[%d]=%f\n", i, c[i]);

	cudaFree(devPtrA);
	cudaFree(devPtrB);
	cudaFree(devPtrC);
	
	return 0;
}
