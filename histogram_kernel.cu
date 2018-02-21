/**
 * @file histogram_kernel.cu
 * @details This file describes the kernel function for a histogram task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#ifndef _HISTOGRAM_KERNEL_H_
#define _HISTOGRAM_KERNEL_H_

#define BIN_COUNT 256
#define HISTOGRAM_SIZE (BIN_COUNT * sizeof(unsigned int))

////////////////////////////////////////////////////////////////////////////////
// GPU-specific definitions
////////////////////////////////////////////////////////////////////////////////
//Machine warp size
#define WARP_LOG2SIZE 5

//Warps in thread block for histogram256Kernel()
#define WARP_N 12

//Corresponding thread block size in threads for histogram256Kernel()
#define THREAD_N (WARP_N << WARP_LOG2SIZE)

//Total histogram size (in counters) per thread block for histogram256Kernel()
#define BLOCK_MEMORY (WARP_N * BIN_COUNT)

//Thread block count for histogram256Kernel()
#define BLOCK_N 64

//#define D_S 528 // Pal - 192 threads/block
//#define D_S 396 // Pal - 256 threads/block
#define D_S 264 // Pal - 384 threads/block
//#define D_S 132 // Half-pal
//#define D_S 33 // Quarter-pal


__device__ inline void addData256(unsigned int *s_WarpHist, unsigned int data){
        atomicAdd(s_WarpHist + data, 1);
}

/**
 * @brief Histogram kernel function.
 * @details CUDA Kernel for the histogram.
 * @author CUDA SDK.
 * 
 * @param d_Result Result vector.
 * @param d_Data Input data vector.
 * @param dataN Amount of elements of the vector.
 */
__global__ void histogram256Kernel(unsigned int *d_Result, unsigned int *d_Data, int dataN){
    //Current global thread index
    const int    globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    //Total number of threads in the compute grid
    const int   numThreads = blockDim.x * gridDim.x;
    const int H = (int)(blockIdx.x / D_S);

    //Shared memory storage for each warp
    __shared__ unsigned int s_Hist[BLOCK_MEMORY];

    //Current warp shared memory base
    const int warpBase = (threadIdx.x >> WARP_LOG2SIZE) * BIN_COUNT;

    //Clear shared memory buffer for current thread block before processing
    for(int pos = threadIdx.x; pos < BLOCK_MEMORY; pos += blockDim.x)
       s_Hist[pos] = 0;

    //Cycle through the entire data set, update subhistograms for each warp
    __syncthreads();

    for(int pos = globalTid; pos < dataN; pos += numThreads){
        unsigned int data4 = d_Data[pos];

        addData256(s_Hist + warpBase, (data4 >>  0) & 0xFFU);
        addData256(s_Hist + warpBase, (data4 >>  8) & 0xFFU);
        addData256(s_Hist + warpBase, (data4 >> 16) & 0xFFU);
        addData256(s_Hist + warpBase, (data4 >> 24) & 0xFFU);
    }

    __syncthreads();
    //Merge per-warp histograms into per-block and write to global memory
    for(int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x){
        unsigned int sum = 0;
        for(int base = 0; base < BLOCK_MEMORY; base += BIN_COUNT)
            sum += s_Hist[base + pos] & 0x07FFFFFFU;
            atomicAdd(d_Result + H*BIN_COUNT + pos, sum);
    }
}



#endif
