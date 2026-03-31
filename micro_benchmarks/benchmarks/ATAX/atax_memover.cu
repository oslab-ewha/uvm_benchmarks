/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include "timer.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
//#define NX 1200
//#define NY 1200
#define NX 45970 //40*1024
#define NY 45970 //40*1024

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
    unsigned long long i, j;

    for (i = 0; i < NX; i++)
    {
        x[i] = i * M_PI;
        for (j = 0; j < NY; j++)
        {
            A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
        }
    }

    printf("initialization is done\n");
}

__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j;

    if (i < NX)
    {
        float acc = 0.f;
        for(j=0; j < NY; j++)
        {
            acc += A[i * NY + j] * x[j];
        }
        tmp[i] = acc;
    }
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long i;

    if (j < NY)
    {
        float acc = 0.f;

        for(i=0; i < NX; i++)
        {
            acc += A[i * NY + j] * tmp[i];
        }
        y[j] = acc;
    }
}

void ataxGpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    //dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
    //dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
    dim3 grid1( (NX + block.x - 1) / block.x, 1 );
    dim3 grid2( (NY + block.x - 1) / block.x, 1 );

    atax_kernel1<<< grid1, block >>>(A,x,tmp);

    atax_kernel2<<< grid2, block >>>(A,y,tmp);

    // Wait for GPU to finish before accessing on host
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


int main(int argc, char** argv)
{
    DATA_TYPE* A;
    DATA_TYPE* x;
    DATA_TYPE* y;
    DATA_TYPE* tmp;
    unsigned long long elapsed;

    //size_t size = 6L * 1024 * 1024 * 1024; //6GB
    size_t size = 6373L * 1024 * 1024; // 6373MiB
    unsigned *c_m = NULL;
    unsigned *d_m = NULL;
    alloc_ext_mem(c_m, d_m, size);

    cudaMallocManaged( &A, (unsigned long long)NX*NY*sizeof(DATA_TYPE));
    cudaMallocManaged( &x, NY*sizeof(DATA_TYPE));
    cudaMallocManaged( &y, NY*sizeof(DATA_TYPE));
    cudaMallocManaged( &tmp, NX*sizeof(DATA_TYPE));

    init_array(x, A);

    init_tickcount();

    ataxGpu(A, x, y, tmp);

// #define TRACEBACK
#ifdef TRACEBACK
    FILE *fp;

    fp = fopen("result_ATAX.txt","a+");

    for(int i = 0; i < NY; i++) {
        fprintf(fp, "%lf\n", y[i]);
    }

    fclose(fp);

#else
    /*for(int i = 0; i < NY; i++) {
        printf(fp, "%lf\n", y[i]);
    }*/

#endif

    cudaFree(A);
    cudaFree(x);
    cudaFree(y);
    cudaFree(tmp);

    elapsed = get_tickcount_us(); //get_tickcount();
    printf("elapsed_time(us): %llu\n", elapsed);

    free_ext_mem(c_m, d_m);

    return 0;
}

