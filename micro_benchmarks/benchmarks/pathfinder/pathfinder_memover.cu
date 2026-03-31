#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "timer.h"

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

void run(int argc, char** argv);

unsigned long long rows, cols;
int* data;
int* gpuWall;
int* gpuResult[2];
#define M_SEED 9
int pyramid_height;

void
init(int argc, char** argv)
{
    if(argc==4){
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
        pyramid_height = atoi(argv[3]);
    }
    else{
        printf("Usage: pathfinder col_len row_len pyramid_height\n");
        exit(0);
    }
    data = new int[rows*cols];

    cudaMallocManaged((void**)&gpuResult[0], sizeof(int) * cols);
    cudaMallocManaged((void**)&gpuResult[1], sizeof(int) * cols);
    cudaMallocManaged((void**)&gpuWall, sizeof(int)*(rows * cols - cols));

    int seed = M_SEED;
    srand(seed);

    for (unsigned long long i = 0; i < rows; i++)
    {
        for (unsigned long long j = 0; j < cols; j++)
        {
            if (i == 0) {
                gpuResult[0][j] = rand() % 10;
                data[i*cols + j] = gpuResult[0][j];
            } else {
                gpuWall[(i-1)*cols + j] = rand() % 10;
                data[i*cols + j] = gpuWall[(i-1)*cols + j];
            }
        }
    }
// #define TRACEBACK
#ifdef TRACEBACK
    for (unsigned long long i = 0; i < rows; i++)
    {
        for (unsigned long long j = 0; j < cols; j++)
        {
            if (i == 0) {
                printf("%d ", gpuResult[0][j]);
            } else {
                printf("%d ", gpuWall[(i-1)*cols + j]);
            }
        }
        printf("\n");
    }
#endif
}

void 
fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                unsigned long long cols, 
                unsigned long long rows,
                int startStep,
                int border)
{

    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // each block finally computes result for a small block
    // after N iterations. 
    // it is the non-overlapping small blocks that cover 
    // all the input data

    // calculate the small block size
    int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

    // calculate the boundary for the block according to 
    // the boundary of its small block
    int blkX = small_block_cols*bx-border;
    int blkXmax = blkX+BLOCK_SIZE-1;

    // calculate the global thread coordination
    int xidx = blkX+tx;

    // effective range within this block that falls within 
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

    int W = tx-1;
    int E = tx+1;

    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if(IN_RANGE(xidx, 0, cols-1)){
        prev[tx] = gpuSrc[xidx];
    }
    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    bool computed;
    for (int i=0; i<iteration ; i++){ 
        computed = false;
        if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid) {
            computed = true;
            int left = prev[W];
            int up = prev[tx];
            int right = prev[E];
            int shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            unsigned long long index = cols*(startStep+i)+xidx;
            result[tx] = shortest + gpuWall[index];

        }
        __syncthreads();
        if(i == iteration-1)
            break;
        if(computed)     //Assign the computation range
            prev[tx]= result[tx];
        __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the 
    // small block perform the calculation and switch on ``computed''
    if (computed){
        gpuResults[xidx]=result[tx];
    }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], \
              unsigned long long rows, unsigned long long cols, \
              int pyramid_height, int blockCols, int borderCols)
{
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);  

#ifdef PREF
    cudaStream_t stream3;
    cudaStreamCreate(&stream3);
#endif

    int src = 1, dst = 0;
    for (int t = 0; t < rows-1; t+=pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
#ifdef PREF
        dynproc_kernel<<<dimGrid, dimBlock, 0, stream3>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);
#else
        dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);
#endif
    }
    return dst;
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
    //int num_devices;
    //cudaGetDeviceCount(&num_devices);
    //if (num_devices > 1) cudaSetDevice(DEVICE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%lld]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
    pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

    int size = rows*cols;
    unsigned long long elapsed;

    size_t m_size =  6599ULL * 1024ULL * 1024ULL; // 6599MiB
    unsigned *c_m = NULL;
    unsigned *d_m = NULL;
    alloc_ext_mem(c_m, d_m, m_size);

    init_tickcount();

#ifdef PREF
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

    cudaMemPrefetchAsync( gpuResult[0], sizeof(int)*cols, DEVICE, stream1);
    cudaMemPrefetchAsync( gpuWall, sizeof(int)*(size-cols), DEVICE, stream2);
#endif

    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);

    cudaDeviceSynchronize();


//#define TRACEBACK
#ifdef TRACEBACK
    for (int i = 0; i < cols; i++)
        printf("%d ",data[i]);
    printf("\n");
    for (int i = 0; i < cols; i++)
        printf("%d ",gpuResult[final_ret][i]);
    printf("\n") ;
#endif

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete [] data;

    elapsed = get_tickcount_us(); //get_tickcount();
    printf("elapsed_time(us): %llu\n", elapsed);

    free_ext_mem(c_m, d_m);
}

