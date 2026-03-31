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
#include "utils.h"
#include "parser.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

//#define GPU_DEVICE 0

/* Problem size */
//#define tmax 500
//#define NX 2048
//#define NY 2048
#define tmax 5
#define NX 32768 //#define NX 1200
#define NY 32768 //#define NY 1200


/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

static void
usage(void)
{
    printf(
            "fdtd2d <options>\n"
            "<options>:\n"
            "  -e <reserve_extra_memory>: whether to reserve extra memory\n"
            "  -h: help\n");
}
static int reserve_extra_memory = 0;


void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
    int i, j;

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
    FILE *fp;
    fp = fopen("result.txt","w");

    for(int i = 0; i < NX*NY; i+= 1000) {
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


// run FDTD on CPU
void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
    int t, i, j;

    for (t=0; t < tmax; t++)  
    {
        for (j=0; j < NY; j++)
        {
            ey[0*NY + j] = _fict_[t];
        }

        for (i = 1; i < NX; i++)
        {
            for (j = 0; j < NY; j++)
            {
                ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
            }
        }

        for (i = 0; i < NX; i++)
        {
            for (j = 1; j < NY; j++)
            {
                ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
            }
        }

        for (i = 0; i < NX; i++)
        {
            for (j = 0; j < NY; j++)
            {
                hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
            }
        }
    }
}


void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
    int i, j, fail;
    fail = 0;

    for (i=0; i < NX; i++) 
    {
        for (j=0; j < NY; j++) 
        {
            if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
            {
                fail++;
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

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
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NX) && (j < NY) && (j > 0))
    {
        ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
    }
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NX) && (j < NY))
    {
        hz[i * NY + j] = hz[i * NY + j]
                         - 0.7f * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
    }
}


void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
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

// parse user input
static void
parse_args(int argc, char *argv[])
{
    int c;

    while ((c = getopt(argc, argv, "eh")) != -1) {
        switch (c) {
            case 'e':
                reserve_extra_memory = 1;
                break;
            case 'h':
                usage();
                exit(0);
            default:
                usage();
                ERROR("invalid argument");
        }
    }
}


int main(int argc, char** argv)
{
    DATA_TYPE* _fict_;
    DATA_TYPE* ex;
    DATA_TYPE* ey;
    DATA_TYPE* hz;
    //DATA_TYPE* hz_CPU;
    unsigned elapsed;

    unsigned *c_m = NULL;
    unsigned *d_m = NULL;

    parse_args(argc, argv);

    if (reserve_extra_memory)
    {
        size_t size = 6373L * 1024 * 1024; // 6373MiB
        alloc_ext_mem(c_m, d_m, size);
    }

    cudaMallocManaged(&_fict_, tmax*sizeof(DATA_TYPE));
    cudaMallocManaged(&ex, (unsigned long long)NX*(NY+1)*sizeof(DATA_TYPE));
    cudaMallocManaged(&ey, (unsigned long long)(NX+1)*NY*sizeof(DATA_TYPE));
    cudaMallocManaged(&hz, (unsigned long long)NX*NY*sizeof(DATA_TYPE));

    //hz_CPU = (DATA_TYPE*)malloc((unsigned long long)NX*NY*sizeof(DATA_TYPE));

    init_arrays(_fict_, ex, ey, hz);

    init_tickcount();

    //GPU_argv_init();
    fdtdCuda(_fict_, ex, ey, hz);

    //runFdtd(_fict_, ex, ey, hz_CPU);
    //compareResults(hz_CPU, hz);

    writeoutput(hz);

    cudaFree(_fict_);
    cudaFree(ex);
    cudaFree(ey);
    cudaFree(hz);
    //free(hz_CPU);

    elapsed = get_tickcount();
    printf("elapsed_time(us): %u\n", elapsed);

    if (reserve_extra_memory)
        free_ext_mem(c_m, d_m);

    return 0;
}

