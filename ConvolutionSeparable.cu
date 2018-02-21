/**
 * @file ConvolutionSeparable.cu
 * @details This file describes the functions belonging to ConvolutionSeparable class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "ConvolutionSeparable.h"
#include "ConvolutionSeparable_kernel.cu"



ConvolutionSeparable::ConvolutionSeparable(int w, int h, int nIter)
{
	imageW     = w;
    imageH     = h;
    iterations = nIter;
	
}


ConvolutionSeparable::~ConvolutionSeparable()
{
	//Free host memory
    if(h_Input     != NULL) cudaFreeHost(h_Input);
    if(h_OutputGPU != NULL) cudaFreeHost(h_OutputGPU); 
    if(h_Kernel    != NULL) delete [] h_Kernel;
    if(h_Buffer    != NULL) delete [] h_Buffer; 
    if(h_OutputCPU != NULL) delete [] h_OutputCPU;

    //Free device memory
    if(d_Input     != NULL) cudaFree(d_Input);
    if(d_Output    != NULL) cudaFree(d_Output);
    if(d_Buffer    != NULL) cudaFree(d_Buffer); 

}

void ConvolutionSeparable::allocHostMemory(void)
{
		
    cudaMallocHost((void **)&h_Input,     imageW * imageH * sizeof(float));
    cudaMallocHost((void **)&h_OutputGPU, imageW * imageH * sizeof(float)); 
    h_Kernel    =  new float [KERNEL_LENGTH];
    h_Buffer    =  new float [imageW * imageH]; 
    h_OutputCPU =  new float [imageW * imageH]; 
    
   
	
	
}

void ConvolutionSeparable::freeHostMemory(void)
{
	if(h_Input     != NULL) cudaFreeHost(h_Input);
    if(h_OutputGPU != NULL) cudaFreeHost(h_OutputGPU); 
    if(h_Kernel    != NULL) delete [] h_Kernel;
    if(h_Buffer    != NULL) delete [] h_Buffer; 
    if(h_OutputCPU != NULL) delete [] h_OutputCPU; 
     
	
	
}

void ConvolutionSeparable::allocDeviceMemory(void)
{
	cudaMalloc((void **)&d_Input,   imageW * imageH * sizeof(float));
    cudaMalloc((void **)&d_Output,  imageW * imageH * sizeof(float));
    cudaMalloc((void **)&d_Buffer , imageW * imageH * sizeof(float));
	
}

void ConvolutionSeparable::freeDeviceMemory(void)
{
    if(d_Input  != NULL) cudaFree(d_Input);
    if(d_Output != NULL) cudaFree(d_Output);
    if(d_Buffer != NULL) cudaFree(d_Buffer);

	
}

void ConvolutionSeparable::generatingData(void)
{
	srand(200);

    for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    {
        h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (float)(rand() % 16);
    }

    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
	
	
}

void ConvolutionSeparable::memHostToDeviceAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice, stream);
	
}

void ConvolutionSeparable::memHostToDevice(void)
{

    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice); 


}

void ConvolutionSeparable::memDeviceToHostAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost, stream);
	
}

void ConvolutionSeparable::memDeviceToHost(void)
{
    cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);
}

void ConvolutionSeparable::launch_kernel_Async(cudaStream_t stream)
{
	
	for (int i = 0; i < iterations; i++)
    {
        
        convolutionRowsGPUAsync(
            d_Buffer,
            d_Input,
            imageW,
            imageH,
            stream
        );

        convolutionColumnsGPUAsync(
            d_Output,
            d_Buffer,
            imageW,
            imageH,
            stream
        );
    }
	
	
}

void ConvolutionSeparable::launch_kernel(void)
{
	for (int i = 0; i < iterations; i++)
    {
        
        convolutionRowsGPU(
            d_Buffer,
            d_Input,
            imageW,
            imageH
        );

        convolutionColumnsGPU(
            d_Output,
            d_Buffer,
            imageW,
            imageH
        );
    }
        


}

void ConvolutionSeparable::checkResults(void)
{
	convolutionRowCPU(
        h_Buffer,
        h_Input,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

   
    convolutionColumnCPU(
        h_OutputCPU,
        h_Buffer,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
        sum   += h_OutputCPU[i] * h_OutputCPU[i];
    }

    L2norm = sqrt(delta / sum);
	
}




void ConvolutionSeparable::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = imageW * imageH * sizeof(float);
	
	
}

void ConvolutionSeparable::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = imageW * imageH * sizeof(float);
	
}

void ConvolutionSeparable::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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

void ConvolutionSeparable::convolutionColumnsGPUAsync(float *d_Dst, float *d_Src, 
    int imageW, int imageH, cudaStream_t stream)
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads, 0, stream>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );
}

void ConvolutionSeparable::convolutionColumnsGPU(float *d_Dst, float *d_Src, 
    int imageW, int imageH)
{
   assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );
}

void ConvolutionSeparable::convolutionRowsGPUAsync(float *d_Dst, float *d_Src, 
    int imageW, int imageH, cudaStream_t stream)
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolutionRowsKernel<<<blocks, threads, 0, stream>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );
}

void ConvolutionSeparable::convolutionRowsGPU(float *d_Dst, float *d_Src, 
    int imageW, int imageH)
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolutionRowsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );
}

void ConvolutionSeparable::convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Kernel, int imageW,
                        int imageH, int kernelR)
{
    for (int y = 0; y < imageH; y++)
        for (int x = 0; x < imageW; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = x + k;

                if (d >= 0 && d < imageW)
                    sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
}

void ConvolutionSeparable::convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Kernel, int imageW,
    int imageH, int kernelR)
{
    for (int y = 0; y < imageH; y++)
        for (int x = 0; x < imageW; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = y + k;

                if (d >= 0 && d < imageH)
                    sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
}

