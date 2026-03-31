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
#define NX 52*1024
#define NY 52*1024
#define M 32

/* Thread block dimensions */
//#define DIM_THREAD_BLOCK_X 256
//#define DIM_THREAD_BLOCK_Y 1
#define DIM_THREAD_BLOCK_X 64
#define DIM_THREAD_BLOCK_Y 16

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
        for (j = 0; j < M; j++)
        {
            x[i * M + j] = i * M_PI;
        }
        for (j = 0; j < NY; j++)
        {
            A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
        }
    }
}

__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
    unsigned long long row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < NX && col < M)
    {
        unsigned long long i;
        unsigned long long temp = 0;

        for(i = 0; i < NY; i++)
        {
            temp += A[row * NY + i] * x[i * M + col];
        }

    tmp[row * M + col] = temp;

    }
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
    unsigned long long col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < NY && col < M)
    {
        unsigned long long i;
        unsigned long long temp = 0;

        for(i = 0; i < NX; i++)
        {
            temp += A[row * NX + i] * tmp[i * M + col];
        }

    y[row * M + col] = temp;

    }

}

void ataxGpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
    dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

    atax_kernel1<<< grid1, block >>>(A,x,tmp);

    //atax_kernel2<<< grid2, block >>>(A,y,tmp);
    atax_kernel2<<< (size_t)(ceil( ((float)NY) / ((float)DIM_THREAD_BLOCK_X) )), DIM_THREAD_BLOCK_X >>>(A,y,tmp);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}


int main(int argc, char** argv)
{
    DATA_TYPE* A;
    DATA_TYPE* x;
    DATA_TYPE* y;
    DATA_TYPE* tmp;
    unsigned elapsed;

    cudaMallocManaged( &A, (unsigned long long)NX*NY*sizeof(DATA_TYPE));
    cudaMallocManaged( &x, (unsigned long long)NY*M*sizeof(DATA_TYPE));
    cudaMallocManaged( &y, (unsigned long long)NY*M*sizeof(DATA_TYPE));
    cudaMallocManaged( &tmp, (unsigned long long)NX*M*sizeof(DATA_TYPE));

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

    elapsed = get_tickcount();
    printf("elapsed_time(us): %u\n", elapsed);

    return 0;
}

