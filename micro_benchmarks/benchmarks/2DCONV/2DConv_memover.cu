/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include "timer.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
// #define NI 1024
// #define NJ 1024
#define NI 60*1024
#define NJ 18016 //32*563

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init(DATA_TYPE* A)
{
    //int i, j;
    unsigned long long i, j;

    for (i = 0; i < NI; ++i)
    {
        for (j = 0; j < NJ; ++j)
        {
            A[i*NJ + j] = (float)rand()/RAND_MAX;
        }
    }
}

__global__ void Convolution2D_kernel(DATA_TYPE *A, DATA_TYPE *B)
{
    //int j = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long i = blockIdx.y * blockDim.y + threadIdx.y;

    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
    c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
    c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

    if ((i < NI-1) && (j < NJ-1) && (i > 0) && (j > 0))
    {
        B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] 
            + c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
            + c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
    }
}


void convolution2DCuda(DATA_TYPE* A, DATA_TYPE* B)
{
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    //dim3 grid((size_t)ceil( ((float)NI) / ((float)block.x) ), (size_t)ceil( ((float)NJ) / ((float)block.y)) );
    dim3 grid( (NJ + block.x - 1) / block.x, (NI + block.y - 1) / block.y );

    Convolution2D_kernel<<<grid, block>>>(A, B);

    // Wait for GPU to finish before accessing on host
    // mock synchronization of memory specific to stream
    cudaDeviceSynchronize();
}

///////////////////////////////////////////////////////////////
//Allocate extra gpu memory (for memory oversubscription test)
///////////////////////////////////////////////////////////////
void alloc_ext_mem(unsigned *c_m, unsigned *d_m, size_t size)
{
    c_m = (unsigned *)malloc((unsigned long long)size);
    cudaMalloc((unsigned**)&d_m, (unsigned long long)size);
    cudaMemcpy(d_m, c_m, (unsigned long long)size, cudaMemcpyHostToDevice);
    //printf("memcpy: %llu\n", (unsigned long long)size);
}
void free_ext_mem(unsigned *c_m, unsigned *d_m)
{
    free(c_m);
    cudaFree(d_m);
}

int main(int argc, char *argv[])
{
    DATA_TYPE* A;
    DATA_TYPE* B;  
    unsigned long long elapsed;

    size_t size =  6424ULL * 1024ULL * 1024ULL; // 6424MiB
    unsigned *c_m = NULL;
    unsigned *d_m = NULL;
    alloc_ext_mem(c_m, d_m, size);

    cudaMallocManaged( &A, NI*NJ*sizeof(DATA_TYPE) );
    cudaMallocManaged( &B, NI*NJ*sizeof(DATA_TYPE) );

    //initialize the arrays
    init(A);

    init_tickcount();

    convolution2DCuda(A, B);

// #define TRACEBACK
#ifdef TRACEBACK
    FILE *fp;

    fp = fopen("result_2DConv.txt","a+");

    for(int i = 0; i < NI*NJ; i+= 10000) {
        fprintf(fp, "%lf\n", B[i]);
    }
    
    fclose(fp);

#else
    /*DATA_TYPE B_c;
    for(i = 0; i < (unsigned long long)NI*NJ; i+= 10000) {
        B_c = B[i];
    }*/

#endif

    cudaFree(A);
    cudaFree(B);

    elapsed = get_tickcount_us(); //get_tickcount();
    printf("elapsed_time(us): %llu\n", elapsed);

    free_ext_mem(c_m, d_m);

    return 0;
}

