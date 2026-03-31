/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>
#include "timer.h"
#include "utils.h"
#include "parser.h"

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Define the possible dataset sizes. */
//#define NX 4096
//#define NY 4096
#define NX 45970 //40*1024
#define NY 45970 //40*1024

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#define DATA_TYPE float

static void
usage(void)
{
    printf(
            "bicg <options>\n"
            "<options>:\n"
            "  -e <reserve_extra_memory>: whether to reserve extra memory\n"
            "  -h: help\n");
}
static int reserve_extra_memory = 0;

void init_array(unsigned long long nx, unsigned long long ny, DATA_TYPE* A, DATA_TYPE* p, DATA_TYPE* r)
{
    unsigned long long i, j;

    for (i = 0; i < ny; i++)
    {
        p[i] = i * M_PI;
    }

    for (i = 0; i < nx; i++)
    {
        r[i] = i * M_PI;

        for (j = 0; j < ny; j++)
        {
            //A[i][j] = ((DATA_TYPE) i*j) / NX;
            A[i*NY + j] = ((DATA_TYPE) i*j) / NX;
        }
    }
}


void compareResults(unsigned long long nx, unsigned long long ny, DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, 
                    DATA_TYPE* q, DATA_TYPE* q_outputFromGpu)
{
    unsigned long long i,fail;
    fail = 0;

    // Compare s with s_cuda
    for (i=0; i<nx; i++)
    {
        if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }
    }

    for (i=0; i<ny; i++)
    {
        if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }
    }

    // print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %llu\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


//Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
__global__ void bicg_kernel1(unsigned long long nx, unsigned long long ny, DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < NY)
    {
        //s[j] = 0.0f;
        float acc = 0.f;

        for(unsigned long long i = 0; i < NX; i++)
        {
            //s[j] += r[i] * A[i * NY + j];
            acc += r[i] * A[i * NY + j];
        }
        s[j] = acc;
    }
}


//Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(unsigned long long nx, unsigned long long ny, DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NX)
    {
        //q[i] = 0.0f;
        float acc = 0.f;

        for(unsigned long long j=0; j < NY; j++)
        {
            //q[i] += A[i * NY + j] * p[j];
            acc += A[i * NY + j] * p[j];
        }
        q[i] = acc;
    }
}


void bicg_cpu(unsigned long long nx, unsigned long long ny, DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{
    unsigned long long i,j;

    for (i = 0; i < NY; i++)
    {
        s[i] = 0.0;
    }

    for (i = 0; i < NX; i++)
    {
        q[i] = 0.0;
        for (j = 0; j < NY; j++)
        {
            //s[j] = s[j] + r[i] * A[i][j];
            //q[i] = q[i] + A[i][j] * p[j];
            s[j] = s[j] + r[i] * A[i*NY + j];
            q[i] = q[i] + A[i*NY + j] * p[j];
        }
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(unsigned long long nx, unsigned long long ny, DATA_TYPE* s, DATA_TYPE* q)
{
    unsigned long long i;

    for (i = 0; i < ny; i++) {
        fprintf (stdout, "%0.2lf ", s[i]);
        if (i % 20 == 0) fprintf (stdout, "\n");
    }
    for (i = 0; i < nx; i++) {
        fprintf (stdout, "%0.2lf ", q[i]);
        if (i % 20 == 0) fprintf (stdout, "\n");
    }
    fprintf (stdout, "\n");
}


void bicgCuda(unsigned long long nx, unsigned long long ny, DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
    //, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q_outputFromGpu)
{
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid1((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
    dim3 grid2((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);

    bicg_kernel1<<< grid1, block >>>(nx, ny, A, r, s);
    cudaDeviceSynchronize();

    bicg_kernel2<<< grid2, block >>>(nx, ny, A, p, q);
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
    unsigned long long nx = NX;
    unsigned long long ny = NY;
    unsigned long long elapsed;

    DATA_TYPE* A;
    DATA_TYPE* r;
    DATA_TYPE* s;
    DATA_TYPE* p;
    DATA_TYPE* q;

    unsigned *c_m = NULL;
    unsigned *d_m = NULL;

    parse_args(argc, argv);

    if (reserve_extra_memory)
    {
        size_t size = 6373ULL * 1024ULL * 1024ULL; // 6373MiB
        alloc_ext_mem(c_m, d_m, size);
    }

    cudaMallocManaged(&A, (unsigned long long)NX*NY*sizeof(DATA_TYPE));
    cudaMallocManaged(&r, sizeof(DATA_TYPE) * NX);
    cudaMallocManaged(&s, sizeof(DATA_TYPE) * NY);
    cudaMallocManaged(&p, sizeof(DATA_TYPE) * NY);
    cudaMallocManaged(&q, sizeof(DATA_TYPE) * NX);

    init_array(nx, ny, A, p, r);

    init_tickcount();

    bicgCuda(nx, ny, A, r, s, p, q);

#ifdef RUN_ON_CPU

    DATA_TYPE* s_cpu;
    DATA_TYPE* q_cpu;
    s_cpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
    q_cpu = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

    bicg_cpu(nx, ny, A, r, s_cpu, p, q_cpu);

    //compareResults(nx, ny, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q), 
    //                POLYBENCH_ARRAY(q_outputFromGpu));
    compareResults(nx, ny, s_cpu, s, q_cpu, q);

    free(s_cpu);
    free(q_cpu);

#endif

//#define TRACEBACK
#ifdef TRACEBACK
    print_array(nx, ny, s, q);
#endif

    cudaFree(A);
    cudaFree(r);
    cudaFree(s);
    cudaFree(p);
    cudaFree(q);

    elapsed = get_tickcount_us(); //get_tickcount();
    printf("elapsed_time(us): %llu\n", elapsed);

    if (reserve_extra_memory)
        free_ext_mem(c_m, d_m);

    return 0;
}
