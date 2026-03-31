#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>
#include "timer.h"
#include "parser.h"

// includes, kernels
#include "needle_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

void usage()
{
    printf(
            "needle <options>\n"
            "<options>:\n"
            "  -d <dimension>: x and y dimensions\n"
            "  -p <penalty>: penalty(positive integer)\n"
            "  -h: help\n"
    );
}
static int dimension = 32;
static int penalty = 10;


void init_array(int *itemsets, int *referrence, int max_rows, int max_cols, int penalty)
{
    unsigned long long i, j;
    for (i = 0; i < max_cols; i++){
        for (j = 0; j < max_rows; j++){
            itemsets[i * (unsigned long long)max_cols + j] = 0;
        }
    }

    printf("Start Needleman-Wunsch\n");

    for (i = 1; i < max_rows; i++){    //please define your own sequence. 
        itemsets[i * max_cols] = rand() % 10 + 1;
    }

    for (j = 1; j < max_cols; j++){    //please define your own sequence.
        itemsets[j] = rand() % 10 + 1;
    }

    for (i = 1; i < max_cols; i++){
        for (j = 1; j < max_rows; j++){
            referrence[i * (unsigned long long)max_cols + j] = blosum62[itemsets[i * (unsigned long long)max_cols]][itemsets[j]];
        }
    }

    for(i = 1; i < max_rows; i++)
        itemsets[i * (unsigned long long)max_cols] = -i * penalty;
    for(j = 1; j < max_cols; j++)
        itemsets[j] = -j * penalty;

}

void writeoutput(int* referrence, int* itemsets, int max_rows, int max_cols, int penalty)
{
    int nw, n, w, traceback;
    unsigned long long i, j;

//#define TRACEBACK
#ifdef TRACEBACK
    FILE *fpo = fopen("result.txt","w");
    fprintf(fpo, "print traceback value GPU:\n");

    for (i = max_rows - 2, j = max_rows - 2; i>=0, j>=0; ){

        if (i == max_rows - 2 && j == max_rows - 2)
            fprintf(fpo, "%d ", itemsets[i * (unsigned long long)max_cols + j]); //print the first element

        if (i == 0 && j == 0)
            break;
        if (i > 0 && j > 0) {
            nw = itemsets[(i - 1) * (unsigned long long)max_cols + j - 1];
            w  = itemsets[i * (unsigned long long)max_cols + j - 1];
            n  = itemsets[(i - 1) * (unsigned long long)max_cols + j];
        }
        else if (i == 0) {
            nw = n = LIMIT;
            w  = itemsets[i * (unsigned long long)max_cols + j - 1];
        }
        else if (j == 0) {
            nw = w = LIMIT;
            n  = itemsets[(i - 1) * (unsigned long long)max_cols + j];
        }
        else {
        }

        //traceback = maximum(nw, w, n);
        int new_nw, new_w, new_n;
        new_nw = nw + referrence[i * (unsigned long long)max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);
        if (traceback == new_nw)
            traceback = nw;
        if (traceback == new_w)
            traceback = w;
        if (traceback == new_n)
            traceback = n;

        fprintf(fpo, "%d ", traceback);

        if (traceback == nw)
            {i--; j--; continue;}

        else if (traceback == w)
            {j--; continue;}

        else if (traceback == n)
            {i--; continue;}

        else
            ;
    }

    fclose(fpo);

#else
    /*for (i = max_rows - 2, j = max_rows - 2; i>=0, j>=0; i -= 100, j -= 100){
        nw = itemsets[i * (unsigned long long)max_cols + j];
        traceback = nw + referrence[i * (unsigned long long)max_cols + j];
    }*/
#endif
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
    int	c;

    while ((c = getopt(argc, argv, "d:p:h")) != -1) {
        switch (c) {
            case 'd':
                dimension = parse_count(optarg, "dimension");
                break;
            case 'p':
                penalty = parse_count(optarg, "penalty");
                break;
            case 'h':
                usage();
                exit(0);
            default:
                usage();
                ERROR("invalid argument");
        }
    }

    // the lengths of the two sequences should be able to divided by 16.
    // And at current stage  max_rows needs to equal max_cols
    if (dimension % 16 != 0){
        fprintf(stderr,"The dimension values must be a multiple of 16\n");
        exit(1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv) 
{
    printf("WG size of kernel = %d \n", BLOCK_SIZE);

    runTest(argc, argv);

    return EXIT_SUCCESS;
}


void runTest(int argc, char** argv) 
{
    int max_rows, max_cols;
    int *itemsets,  *referrence;
    unsigned long long size;
    unsigned long long elapsed;

    parse_args(argc, argv);

    size_t m_size = 6339L * 1024 * 1024; // 6339MiB
    unsigned *c_m = NULL;
    unsigned *d_m = NULL;
    alloc_ext_mem(c_m, d_m, m_size);

    max_rows = dimension + 1;
    max_cols = dimension + 1;

    size = max_cols * max_rows;

    cudaMallocManaged(&referrence, sizeof(int)*size);
    cudaMallocManaged(&itemsets, sizeof(int)*size);

    if (!itemsets)
        fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );

    init_array(itemsets, referrence, max_rows, max_cols, penalty);

    init_tickcount();

#ifdef PREF
    int device = -1;
    cudaGetDevice(&device);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

    cudaStream_t stream3;
    cudaStreamCreate(&stream3);

    cudaMemPrefetchAsync(referrence, sizeof(int) * size, device, stream1);
    cudaMemPrefetchAsync(itemsets, sizeof(int) * size, device, stream2);
#endif

    dim3 dimGrid;
    dim3 dimBlock(BLOCK_SIZE, 1);
    int block_width = (max_cols - 1) / BLOCK_SIZE;

    printf("Processing top-left matrix\n");

    //process top-left matrix
    for(int i = 1; i <= block_width; i++){
        dimGrid.x = i;
        dimGrid.y = 1;
#ifdef PREF
        needle_cuda_shared_1<<<dimGrid, dimBlock, 0, stream3>>>(referrence, itemsets, max_cols, penalty, i, block_width); 
#else
        needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence, itemsets, max_cols, penalty, i, block_width); 
#endif
    }

    printf("Processing bottom-right matrix\n");

    //process bottom-right matrix
    for(int i = block_width - 1; i >= 1; i--){
        dimGrid.x = i;
        dimGrid.y = 1;
#ifdef PREF
        needle_cuda_shared_2<<<dimGrid, dimBlock, 0, stream3>>>(referrence, itemsets, max_cols, penalty, i, block_width); 
#else
        needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence, itemsets, max_cols, penalty, i, block_width);
#endif
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    writeoutput(referrence, itemsets, max_rows, max_cols, penalty);

    cudaFree(referrence);
    cudaFree(itemsets);

    elapsed = get_tickcount_us(); //get_tickcount();
    printf("elapsed_time(us): %llu\n", elapsed);

    free_ext_mem(c_m, d_m);

}

