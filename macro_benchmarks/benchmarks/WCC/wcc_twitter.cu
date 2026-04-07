// WCC_Twitter.cu
// Weakly Connected Components(WCC) on soc-twitter-2010 dataset
//


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "timer.h"

#define GPU_DEVICE 0

/* Twitter 2010 graph scale */
#define NUM_NODES  41652230LL       // ~41.6M nodes
#define NUM_EDGES  1468364884LL     // ~1.47B edges


/* Thread block size */
#define DIM_THREAD_BLOCK_X 256


/* -------------------------------------------------------
 * Oversubscription memory
 * ------------------------------------------------------- */
void alloc_ext_mem(unsigned **c_m, unsigned **d_m, size_t size)
{
    *c_m = (unsigned *)malloc((unsigned long long)size);
    cudaMalloc((void**)d_m, (unsigned long long)size);
    cudaMemcpy(*d_m, *c_m, (unsigned long long)size, cudaMemcpyHostToDevice);
}


void free_ext_mem(unsigned *c_m, unsigned *d_m)
{
    free(c_m);
    cudaFree(d_m);
}


/* -------------------------------------------------------
 * WCC kernel: Shiloach-Vishkin label propagation
 * ------------------------------------------------------- */
__global__ void wcc_kernel(
    const int  *src,        // edge source array      [NUM_EDGES]
    const int  *dst,        // edge destination array [NUM_EDGES]
    int        *label,      // component label        [NUM_NODES]
    int        *changed,    // convergence flag (0 or 1)
    long long   num_edges)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    int u = src[tid];
    int v = dst[tid];

    int lu = label[u];
    int lv = label[v];

    if (lu != lv) {
        int mn = (lu < lv) ? lu : lv;
        // write min label to both sides
        // atomicMin: race condition 없이 안전하게 최솟값 갱신
        atomicMin(&label[u], mn);
        atomicMin(&label[v], mn);
        *changed = 1; 
    }
}

/* -------------------------------------------------------
 * Load edge list from CSV 
 * ------------------------------------------------------- */
// load_edgelist(dataset, src, dst, NUM_EDGES);
void load_edgelist(const char *filename, int *h_src, int *h_dst, long long num_edges)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", filename);
        exit(1);
    }

    char line[128];
    long long cnt = 0;
    while (fgets(line, sizeof(line), fp) && cnt < num_edges) {
        // '#' 로 시작하는 주석 줄 스킵 
        if (line[0] == '#') continue;
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) == 2) {
            h_src[cnt] = u;
            h_dst[cnt] = v;
            cnt++;
        }
    }
    fclose(fp);

    if (cnt != num_edges) {
        fprintf(stderr, "WARNING: loaded %lld edges (expected %lld)\n",
                cnt, num_edges);
    }
    printf("Loaded %lld edges from %s\n", cnt, filename);
}

/* -------------------------------------------------------
 * main
 * ------------------------------------------------------- */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path/to/soc-twitter-2010.csv>\n", argv[0]);
        return 1;
    }
    const char *dataset = argv[1];

    cudaSetDevice(GPU_DEVICE);


    // 1. Oversubscription: 6.3GB GPU Memory allocation 
    // size_t ext_size = 6424ULL * 1024ULL * 1024ULL;  // 6424 MiB
    // unsigned *c_m = NULL;
    // unsigned *d_m = NULL;
    // alloc_ext_mem(&c_m, &d_m, ext_size);
    // printf("Oversubscription alloc done: %.1f GB\n",
    //        (double)ext_size / (1024*1024*1024));

    /* --------------------------------------------------
     * 2. cudaMallocManaged for soc-twitter-2010 data 
     *    - src/dst  : 엣지 리스트   (~11 GB, int32 x 2 x 1.47B)
     *    - label    : 노드 레이블   (~166 MB, int32 x 41.6M)
     *    - changed  : 수렴 플래그  (4 bytes)
     * -------------------------------------------------- */
    int *src, *dst, *label, *changed;

    printf("Allocating UVM memory...\n");
    cudaMallocManaged(&src,     NUM_EDGES * sizeof(int)); // 5.5GB
    cudaMallocManaged(&dst,     NUM_EDGES * sizeof(int)); // 5.5GB
    cudaMallocManaged(&label,   NUM_NODES * sizeof(int)); // 166MB
    cudaMallocManaged(&changed, sizeof(int)); // 4Bytes 

    printf("  src+dst  : %.2f GB\n", 2.0 * NUM_EDGES * sizeof(int) / (1024.0*1024*1024));
           // → "  src+dst  : 10.93 GB"
    printf("  label    : %.2f MB\n", 1.0 * NUM_NODES * sizeof(int) / (1024.0*1024));
           // → "  label    : 158.79 MB"

    /* --------------------------------------------------
     * 3. Data Load (data in CPU)
     * -------------------------------------------------- */
    load_edgelist(dataset, src, dst, NUM_EDGES);

    /* label initialization: label[i] = i  */
    for (long long i = 0; i < NUM_NODES; i++)
        label[i] = (int)i;

    /* --------------------------------------------------
     * 4. 커널 실행 설정
     * -------------------------------------------------- */
    dim3 block(DIM_THREAD_BLOCK_X); // #define DIM_THREAD_BLOCK_X 256
    dim3 grid((NUM_EDGES + block.x - 1) / block.x);

    printf("Grid: %u blocks x %d threads\n", grid.x, DIM_THREAD_BLOCK_X);


    init_tickcount();


    // 6. WCC execution 

    int iter = 0;
    do {
        *changed = 0;

        wcc_kernel<<<grid, block>>>(src, dst, label, changed, NUM_EDGES);

        cudaDeviceSynchronize();
        iter++;

        printf("  iter %d: changed=%d\n", iter, *changed);

    } while (*changed); 

    printf("WCC converged in %d iterations\n", iter);



// #define TRACEBACK
#ifdef TRACEBACK
    FILE *fp = fopen("result_WCC.txt", "a+");
    // 10000 간격으로 샘플링 (2DConv 동일)
    for (long long i = 0; i < NUM_NODES; i += 10000)
        fprintf(fp, "label[%lld] = %d\n", i, label[i]);
    fclose(fp);
#endif

    cudaFree(src);
    cudaFree(dst);
    cudaFree(label);
    cudaFree(changed);

    unsigned long long elapsed = get_tickcount_us();
    printf("elapsed_time(us): %llu\n", elapsed);

    // free_ext_mem(c_m, d_m);

    return 0;
}
