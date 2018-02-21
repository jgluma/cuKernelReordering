/**
 * @file SobolQRNG.cu
 * @details This file describes the functions belonging to SobolQRNG class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "SobolQRNG.h"
#include "SobolQRNG_kernel.cu"

SobolQRNG::SobolQRNG(int n_v, int n_d, int gpu)
{
	n_vectors = n_v;
    n_dimensions = n_d;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpu);

    // This implementation of the generator outputs all the draws for
    // one dimension in a contiguous region of memory, followed by the
    // next dimension and so on.
    // Therefore all threads within a block will be processing different
    // vectors from the same dimension. As a result we want the total
    // number of blocks to be a multiple of the number of dimensions.
    n_blocks = n_dimensions;

    // If the number of dimensions is large then we will set the number
    // of blocks to equal the number of dimensions (i.e. dimGrid.x = 1)
    // but if the number of dimensions is small (e.g. less than four per
    // multiprocessor) then we'll partition the vectors across blocks
    // (as well as threads).
    if (n_dimensions < (4 * props.multiProcessorCount))
    {
        n_blocks = 4 * props.multiProcessorCount;
    }
    else
    {
        n_blocks = 1;
    }

    // Cap the dimGrid.x if the number of vectors is small
    if (n_blocks > (unsigned int)(n_vectors / threadsperblock))
    {
        n_blocks = (n_vectors + threadsperblock - 1) / threadsperblock;
    }

    // Round up to a power of two, required for the algorithm so that
    // stride is a power of two.
    unsigned int targetDimGridX = n_blocks;

    for (n_blocks = 1 ; n_blocks < targetDimGridX ; n_blocks *= 2);	
}


SobolQRNG::~SobolQRNG()
{
	if(h_directions != NULL) cudaFreeHost(h_directions);
    if(h_outputGPU != NULL) cudaFreeHost(h_outputGPU);
    if(h_outputCPU != NULL) delete [] h_outputCPU;

    if(d_directions != NULL) cudaFree(d_directions);
    if(d_output != NULL) cudaFree(d_output);
}

void SobolQRNG::allocHostMemory(void)
{
	cudaMallocHost((void **)&h_directions, n_dimensions * n_directions_SOBOLQRNG * sizeof(unsigned int));
    cudaMallocHost((void **)&h_outputGPU, n_vectors * n_dimensions * sizeof(float));
    h_outputCPU  = new float [n_vectors * n_dimensions];
}

void SobolQRNG::freeHostMemory(void)
{
	if(h_directions != NULL) cudaFreeHost(h_directions);
    if(h_outputGPU != NULL) cudaFreeHost(h_outputGPU);
    if(h_outputCPU != NULL) delete [] h_outputCPU;
}

void SobolQRNG::allocDeviceMemory(void)
{
	cudaMalloc((void **)&d_directions, n_dimensions * n_directions_SOBOLQRNG * sizeof(unsigned int));
    cudaMalloc((void **)&d_output, n_vectors * n_dimensions * sizeof(float));
}

void SobolQRNG::freeDeviceMemory(void)
{
	if(d_directions != NULL) cudaFree(d_directions);
    if(d_output != NULL) cudaFree(d_output);
}

void SobolQRNG::generatingData(void)
{
	unsigned int *v = h_directions;

    for (int dim = 0 ; dim < n_dimensions ; dim++)
    {
        // First dimension is a special case
        if (dim == 0)
        {
            for (int i = 0 ; i < n_directions_SOBOLQRNG ; i++)
            {
                // All m's are 1
                v[i] = 1 << (31 - i);
            }
        }
        else
        {
            int d = sobol_primitives[dim].degree;

            // The first direction numbers (up to the degree of the polynomial)
            // are simply v[i] = m[i] / 2^i (stored in Q0.32 format)
            for (int i = 0 ; i < d ; i++)
            {
                v[i] = sobol_primitives[dim].m[i] << (31 - i);
            }

            // The remaining direction numbers are computed as described in
            // the Bratley and Fox paper.
            // v[i] = a[1]v[i-1] ^ a[2]v[i-2] ^ ... ^ a[v-1]v[i-d+1] ^ v[i-d] ^ v[i-d]/2^d
            for (int i = d ; i < n_directions_SOBOLQRNG ; i++)
            {
                // First do the v[i-d] ^ v[i-d]/2^d part
                v[i] = v[i - d] ^ (v[i - d] >> d);

                // Now do the a[1]v[i-1] ^ a[2]v[i-2] ^ ... part
                // Note that the coefficients a[] are zero or one and for compactness in
                // the input tables they are stored as bits of a single integer. To extract
                // the relevant bit we use right shift and mask with 1.
                // For example, for a 10 degree polynomial there are ten useful bits in a,
                // so to get a[2] we need to right shift 7 times (to get the 8th bit into
                // the LSB) and then mask with 1.
                for (int j = 1 ; j < d ; j++)
                {
                    v[i] ^= (((sobol_primitives[dim].a >> (d - 1 - j)) & 1) * v[i - j]);
                }
            }
        }

        v += n_directions_SOBOLQRNG;
    }

    

}

void SobolQRNG::memHostToDeviceAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(d_directions, h_directions, 
        n_dimensions * n_directions_SOBOLQRNG * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
}

void SobolQRNG::memHostToDevice(void)
{
   cudaMemcpy(d_directions, h_directions, 
        n_dimensions * n_directions_SOBOLQRNG * sizeof(unsigned int), cudaMemcpyHostToDevice);
}

void SobolQRNG::memDeviceToHostAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(h_outputGPU, d_output, 
        n_vectors * n_dimensions * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

void SobolQRNG::memDeviceToHost(void)
{
   cudaMemcpy(h_outputGPU, d_output, 
        n_vectors * n_dimensions * sizeof(float), cudaMemcpyDeviceToHost);
}

void SobolQRNG::launch_kernel_Async(cudaStream_t stream)
{
	

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimBlock.x = threadsperblock;
    dimGrid.x = n_blocks;

    sobolGPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(n_vectors, n_dimensions, d_directions, d_output);


}

void SobolQRNG::launch_kernel(void)
{
	// Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimBlock.x = threadsperblock;
    dimGrid.x = n_blocks;

    sobolGPU_kernel<<<dimGrid, dimBlock>>>(n_vectors, n_dimensions, d_directions, d_output);
        
}

void SobolQRNG::checkResults(void)
{
	sobolCPU(n_vectors, n_dimensions, h_directions, h_outputCPU);

    if (n_vectors == 1)
    {
        for (int d = 0, v = 0 ; d < n_dimensions ; d++)
        {
            float ref = h_outputCPU[d * n_vectors + v];
            l1norm_diff += fabs(h_outputGPU[d * n_vectors + v] - ref);
            l1norm_ref  += fabs(ref);
        }

        // Output the L1-Error
        l1error = l1norm_diff;

       
    }
    else
    {
        for (int d = 0 ; d < n_dimensions ; d++)
        {
            for (int v = 0 ; v < n_vectors ; v++)
            {
                float ref = h_outputCPU[d * n_vectors + v];
                l1norm_diff += fabs(h_outputGPU[d * n_vectors + v] - ref);
                l1norm_ref  += fabs(ref);
            }
        }

        // Output the L1-Error
        l1error = l1norm_diff / l1norm_ref;

        
    }


}

void SobolQRNG::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = n_dimensions * n_directions_SOBOLQRNG * sizeof(unsigned int);
	
	
}

void SobolQRNG::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = n_vectors * n_dimensions * sizeof(float);
	
	
}

void SobolQRNG::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
								float *estimated_overlapped_time_HTD, float *estimated_overlapped_time_DTH, 
								float LoHTD, float LoDTH, float GHTD, float GDTH, float overlappedGHTD, float overlappedGDTH)
{
	
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, gpu);

	int bytes_HTD;
	int bytes_DTH;

	getBytesHTD(&bytes_HTD);
	getBytesDTH(&bytes_DTH);
	
	
			
	*estimated_time_HTD = LoHTD + (bytes_HTD) * GHTD;
				
	*estimated_overlapped_time_HTD = 0.0;
		
	if(props.asyncEngineCount == 2)
		*estimated_overlapped_time_HTD = LoHTD + (bytes_HTD) * overlappedGHTD;
			
		
	*estimated_time_DTH = LoDTH + (bytes_DTH) * GDTH;
				
	*estimated_overlapped_time_DTH= 0.0;

		
	if(props.asyncEngineCount == 2)
		*estimated_overlapped_time_DTH= LoDTH + (bytes_DTH) * overlappedGDTH;

	
	
}

void SobolQRNG::sobolCPU(int n_vectors, int n_dimensions, unsigned int *directions, float *output)
{
    unsigned int *v = directions;

    for (int d = 0 ; d < n_dimensions ; d++)
    {
        unsigned int X = 0;
        // x[0] is zero (in all dimensions)
        output[n_vectors * d] = 0.0;

        for (int i = 1 ; i < n_vectors ; i++)
        {
            // x[i] = x[i-1] ^ v[c]
            //  where c is the index of the rightmost zero bit in i
            //  minus 1 (since C arrays count from zero)
            // In the Bratley and Fox paper this is equation (**)
            X ^= v[ffs(~(i - 1)) - 1];
            output[i + n_vectors * d] = (float)X * k_2powneg32_SOBOLQRNG;
        }

        v += n_directions_SOBOLQRNG;
    }
}

