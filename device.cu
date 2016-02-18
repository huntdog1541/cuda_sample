#include <cuda.h>
#include <stdio.h>

int main(void)
{
	int            count;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&count);

	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf ("Device Profile for Device %d\n\n", i);
		printf ("General Information - \n");
		printf (" Name:\t\t\t %s\n", prop.name);
		printf (" Compute capabilities:\t %d.%d\n", prop.major, prop.minor);
		printf (" Clock rate:\t\t %d\n\n", prop.clockRate);

		printf ("Memory Information - \n");
		printf (" Total global memory: \t %ld\n", prop.totalGlobalMem);
		printf (" Total constant memory: %ld\n\n", prop.totalConstMem);

		printf ("Multiprocessor Information - \n");
		printf (" Multiprocessor count:\t %d\n", prop.multiProcessorCount);
		printf (" Shared mem per mp: \t %ld\n", prop.sharedMemPerBlock);
		printf ("Max threads per block: \t %d\n", prop.maxThreadsPerBlock);

		printf ("\n\n");
	}

	return 0;
}
