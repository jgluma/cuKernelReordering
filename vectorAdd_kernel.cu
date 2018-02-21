/**
 * @file vectorAdd_kernel.cu
 * @details This file describes the kernel function for a vector addition task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#ifndef _VECTORADD_KERNEL_H_
#define _VECTORADD_KERNEL_H_
/**
 * @brief Kernel Vector Addition
 * @details CUDA Kernel for the vector addition.
 * @author Antonio Jose Lazaro Muñoz
 * @date 18/02/2016
 * @param A Vector A.
 * @param B Vector B.
 * @param C Vector C.
 * @param numElements Number of elementos of A and B vectors.
 * @param idx_stream CUDA Stream index.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements, int idx_stream)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int idx_global = idx_stream * gridDim.x * blockDim.x + i;
	
	if(idx_global < numElements)
	{
		C[i] = A[i] + B[i];
		
	}
	
}

#endif
