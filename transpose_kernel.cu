/**
 * @file tranpose_kernel.cu
 * @details This file describes the kernel function for a Matrix Transposition task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#ifndef _TRANSPOSE_KERNEL_H_
#define _TRANSPOSE_KERNEL_H_

#define TILE_DIM    16
#define BLOCK_ROWS  16

/**
 * @brief Kernel Matrix Transposition.
 * @details CUDA Kernel for the matrix transposition.
 * @author NVIDIA CUDA SDK

 * @param odata Ouput data vector.
 * @param idata Input data vector.
 * @param width Matrix width.
 * @param height Matrix height.
 * @param nreps Number of repetitions.
 */
__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height, int nreps)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int r=0; r < nreps; r++)
    {
        for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
        {
            tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
        }

        __syncthreads();

        for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
        {
            odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
        }
    }
}

#endif
