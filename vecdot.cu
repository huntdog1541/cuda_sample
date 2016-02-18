#include <cuda.h>
#include <stdio.h>

#define iMin(a, b) (a<b?a:b)

const int N = 17*1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = iMin(16, (N+threadsPerBlock-1)/threadsPerBlock);

// kernel code for adding two vector elements
__global__ void vecDot(float* a, float* b, float* c)
{
	__shared__ float cache[threadsPerBlock];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp=0;
	while (tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	// write partial sum of products into cache
	cache[cacheIndex] = temp;

	// synchronize threads in block
	__syncthreads();

	// reduction of vector across threads in block
	int i=blockDim.x/2;

	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex+i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main(void)
{
	int   i;
	float *a, *b, c, *cPartial;
	float *devPtrA, *devPtrB, *devPtrCPartial;

	// allocate memory for large vectors
	a = (float*) malloc(N*sizeof(float));
	b = (float*) malloc(N*sizeof(float));
	cPartial = (float*) malloc(blocksPerGrid*sizeof(float));

	// initialize arrays
	for (i=0; i < N; i++) {
		a[i] = 1;
		b[i] = 2;
	}

	// allocate CUDA memory for arrays
	int memsize = N*sizeof(float);
	int memsizePartial = blocksPerGrid*sizeof(float);
	cudaMalloc((void**)&devPtrA, memsize);
	cudaMalloc((void**)&devPtrB, memsize);
	cudaMalloc((void**)&devPtrCPartial, memsizePartial);

	// copy host data to CUDA memory
	cudaMemcpy(devPtrA, a, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(devPtrB, b, memsize, cudaMemcpyHostToDevice);

	// call add function on CUDA GPU
	vecDot<<<blocksPerGrid, threadsPerBlock>>>(devPtrA, devPtrB, devPtrCPartial);

	// copy results back
	cudaMemcpy(cPartial, devPtrCPartial, memsizePartial, cudaMemcpyDeviceToHost);

	// compute final result
	c = 0;
	for (i=0; i < blocksPerGrid; i++)
		c += cPartial[i];

	printf("a*b = %f\n", c);

	cudaFree(devPtrA);
	cudaFree(devPtrB);
	cudaFree(devPtrCPartial);

	free (a);
	free (b);
	free (cPartial);
	
	return 0;
}
