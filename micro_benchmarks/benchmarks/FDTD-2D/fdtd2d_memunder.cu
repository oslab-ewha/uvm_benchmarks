/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

/* Problem size */
//#define tmax 500
//#define NX 2048
//#define NY 2048
#define tmax 5
#define NX 36144 //8*4518
#define NY 20416 //32*638

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
    unsigned long long i, j;

    for (i = 0; i < tmax; i++)
    {
        _fict_[i] = (DATA_TYPE) i;
    }

    for (i = 0; i < NX; i++)
    {
        for (j = 0; j < NY; j++)
        {
            ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
            ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
            hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
        }
    }
}

void writeoutput(DATA_TYPE* hz) {
// #define TRACEBACK
#ifdef TRACEBACK
    printf("-------------Size: %lf--------------\n", 
        (float)(sizeof(DATA_TYPE)*(tmax + (unsigned long long)NX*(NY+1) + (unsigned long long)(NX+1)*NY + (unsigned long long)NX*NY))/1024.0/1024.0);

    FILE *fp;
    fp = fopen("result.txt","w");

    for(unsigned long long i = 0; i < NX*NY; i+= 1000) {
        fprintf(fp, "%lf\n", hz[i]);
    }

    fclose(fp);

#else
    /*
    printf("-------------Size: %lf--------------\n", 
        (float)(sizeof(DATA_TYPE)*(tmax + (unsigned long long)NX*(NY+1) + (unsigned long long)(NX+1)*NY + (unsigned long long)NX*NY))/1024.0/1024.0);

    DATA_TYPE hz_c;
    for(unsigned long long i = 0; i < NX*NY; i += 1000) {
        hz_c = hz[i];
    }
    */

#endif
}


/*
void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
    cudaSetDevice( GPU_DEVICE );
}
*/


__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NX) && (j < NY))
    {
        if (i == 0) 
        {
            ey[i * NY + j] = _fict_[t];
        }
        else
        {
            ey[i * NY + j] = ey[i * NY + j] - 0.5f * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
        }
    }
}


__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NX) && (j < NY) && (j > 0))
    {
        ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
    }
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NX) && (j < NY))
    {
        hz[i * NY + j] = hz[i * NY + j]
                         - 0.7f * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
    }
}


void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)//, DATA_TYPE* hz_outputFromGpu)
{
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

    for(int t = 0; t < tmax; t++)
    {
        fdtd_step1_kernel<<<grid,block>>>(_fict_, ex, ey, hz, t);
        cudaDeviceSynchronize();

        fdtd_step2_kernel<<<grid,block>>>(ex, ey, hz, t);
        cudaDeviceSynchronize();

        fdtd_step3_kernel<<<grid,block>>>(ex, ey, hz, t);
        cudaDeviceSynchronize();
    }
}


int main()
{
    DATA_TYPE* _fict_;
    DATA_TYPE* ex;
    DATA_TYPE* ey;
    DATA_TYPE* hz;
    //DATA_TYPE* hz_CPU;
    unsigned long long elapsed;

    cudaMallocManaged(&_fict_, tmax*sizeof(DATA_TYPE));
    cudaMallocManaged(&ex, (unsigned long long)NX*(NY+1)*sizeof(DATA_TYPE));
    cudaMallocManaged(&ey, (unsigned long long)(NX+1)*NY*sizeof(DATA_TYPE));
    cudaMallocManaged(&hz, (unsigned long long)NX*NY*sizeof(DATA_TYPE));

    //hz_CPU = (DATA_TYPE*)malloc((unsigned long long)NX*NY*sizeof(DATA_TYPE));

    init_arrays(_fict_, ex, ey, hz);

    init_tickcount();

    //GPU_argv_init();
    fdtdCuda(_fict_, ex, ey, hz);

    writeoutput(hz);

    cudaFree(_fict_);
    cudaFree(ex);
    cudaFree(ey);
    cudaFree(hz);
    //free(hz_CPU);

    elapsed = get_tickcount_us(); //get_tickcount();
    printf("elapsed_time(us): %llu\n", elapsed);

    return 0;
}

