/**
 * @file FastWalshTransform.cu
 * @details This file describes the functions belonging to FastWalshTransform class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "FastWalshTransform.h"
#include "FastWalshTransform_kernel.cu"

FastWalshTransform::FastWalshTransform(int lg2Data, int lg2Kernel)
{
	
    log2Kernel  = lg2Kernel;
    log2Data    = lg2Data;

    dataN       = 1 << log2Data;
    kernelN     = 1 << log2Kernel;

    DATA_SIZE   = dataN   * sizeof(float);
    KERNEL_SIZE = kernelN * sizeof(float);

	
}


FastWalshTransform::~FastWalshTransform()
{
	//Free host memory
	if(h_Kernel      != NULL) cudaFreeHost(h_Kernel);
    if(h_Data        != NULL) cudaFreeHost(h_Data);
    if(h_ResultGPU   != NULL) cudaFreeHost(h_ResultGPU);
    if(h_ResultCPU   != NULL) free(h_ResultCPU);
    if(h_Kernel_zero != NULL) free(h_Kernel_zero);

	//Free device memory
    if(d_Kernel      != NULL) cudaFree(d_Kernel);
    if(d_Data        != NULL) cudaFree(d_Data);
	

    	

}

void FastWalshTransform::allocHostMemory(void)
{
		
    cudaMallocHost((void **)&h_Kernel, KERNEL_SIZE);
    cudaMallocHost((void **)&h_Data, DATA_SIZE);
    cudaMallocHost((void **)&h_ResultGPU, DATA_SIZE);
    cudaMallocHost((void **)&h_Kernel_zero, DATA_SIZE);
    h_ResultCPU   = (float *)malloc(DATA_SIZE);
    
	
	
}

void FastWalshTransform::freeHostMemory(void)
{
	
	//Free host memory
    if(h_Kernel      != NULL) cudaFreeHost(h_Kernel);
    if(h_Data        != NULL) cudaFreeHost(h_Data);
    if(h_ResultGPU   != NULL) cudaFreeHost(h_ResultGPU);
    if(h_ResultCPU   != NULL) free(h_ResultCPU);
    if(h_Kernel_zero != NULL) cudaFreeHost(h_Kernel_zero);
	
}

void FastWalshTransform::allocDeviceMemory(void)
{
	
	cudaMalloc((void **)&d_Kernel, DATA_SIZE);
    cudaMalloc((void **)&d_Data,   DATA_SIZE);
	
	
}

void FastWalshTransform::freeDeviceMemory(void)
{
	
	//Free device memory
	if(d_Kernel != NULL) cudaFree(d_Kernel);
    if(d_Data   != NULL) cudaFree(d_Data);

}

void FastWalshTransform::generatingData(void)
{
	
	srand(2007);
    memset(h_Kernel_zero, 0, DATA_SIZE);

    for (int i = 0; i < kernelN; i++)
    {
        h_Kernel[i] = (float)rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < dataN; i++)
    {
        h_Data[i] = (float)rand() / (float)RAND_MAX;
    }
	
}

void FastWalshTransform::memHostToDeviceAsync(cudaStream_t stream)
{
	
	 cudaMemcpyAsync(d_Kernel, h_Kernel_zero, DATA_SIZE,   cudaMemcpyHostToDevice, stream);
     cudaMemcpyAsync(d_Kernel, h_Kernel,      KERNEL_SIZE, cudaMemcpyHostToDevice, stream);
     cudaMemcpyAsync(d_Data,   h_Data,        DATA_SIZE,   cudaMemcpyHostToDevice, stream);
	
}

void FastWalshTransform::memHostToDevice(void)
{

     cudaMemcpy(d_Kernel, h_Kernel_zero, DATA_SIZE,   cudaMemcpyHostToDevice);
     cudaMemcpy(d_Kernel, h_Kernel,      KERNEL_SIZE, cudaMemcpyHostToDevice);
     cudaMemcpy(d_Data,   h_Data,        DATA_SIZE,   cudaMemcpyHostToDevice);


}

void FastWalshTransform::memDeviceToHostAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost, stream);
}

void FastWalshTransform::memDeviceToHost(void)
{
    cudaMemcpy(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost);
}

void FastWalshTransform::launch_kernel_Async(cudaStream_t stream)
{
	
	fwtBatchGPUAsync(d_Data,   1,        log2Data, stream);
    fwtBatchGPUAsync(d_Kernel, 1,        log2Data, stream);
    modulateGPUAsync(d_Data,   d_Kernel, dataN,    stream);
    fwtBatchGPUAsync(d_Data,   1,        log2Data, stream);
	
	
}

void FastWalshTransform::launch_kernel(void)
{
	fwtBatchGPU(d_Data,   1,        log2Data);
    fwtBatchGPU(d_Kernel, 1,        log2Data);
    modulateGPU(d_Data,   d_Kernel, dataN);
    fwtBatchGPU(d_Data,   1,        log2Data);
}

void FastWalshTransform::checkResults(void)
{
	
	dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);
	
    sum_delta2 = 0;
    sum_ref2   = 0;

    for (int i = 0; i < dataN; i++)
    {
        delta       = h_ResultCPU[i] - h_ResultGPU[i];
        ref         = h_ResultCPU[i];
        sum_delta2 += delta * delta;
        sum_ref2   += ref * ref;
    }

    L2norm = sqrt(sum_delta2 / sum_ref2);
	
}

void FastWalshTransform::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = KERNEL_SIZE + DATA_SIZE + DATA_SIZE;
	
	
}

void FastWalshTransform::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = DATA_SIZE;
	
	
}

void FastWalshTransform::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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


void FastWalshTransform::fwtBatchGPUAsync(float *d_Data, int M, int log2N, cudaStream_t stream)
{
    const int THREAD_N = 256;

    int N = 1 << log2N;
    dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);

    for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
    {
        fwtBatch2Kernel<<<grid, THREAD_N, 0, stream>>>(d_Data, d_Data, N / 4);
        
    }

    fwtBatch1Kernel<<<M, N / 4, N *sizeof(float), stream>>>(
        d_Data,
        d_Data,
        log2N
    );
}

void FastWalshTransform::fwtBatchGPU(float *d_Data, int M, int log2N)
{
    const int THREAD_N = 256;

    int N = 1 << log2N;
    dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);

    for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
    {
        fwtBatch2Kernel<<<grid, THREAD_N>>>(d_Data, d_Data, N / 4);
        
    }

    fwtBatch1Kernel<<<M, N / 4, N *sizeof(float)>>>(
        d_Data,
        d_Data,
        log2N
    );
}

//Interface to modulateKernel()
void FastWalshTransform::modulateGPUAsync(float *d_A, float *d_B, int N, cudaStream_t stream)
{
    modulateKernel<<<128, 256, 0, stream>>>(d_A, d_B, N);
}

void FastWalshTransform::modulateGPU(float *d_A, float *d_B, int N)
{
    modulateKernel<<<128, 256>>>(d_A, d_B, N);
}

void FastWalshTransform::dyadicConvolutionCPU(float *h_Result, float *h_Data, float *h_Kernel, 
        int log2dataN, int log2kernelN)
{
    for (int i = 0; i < dataN; i++)
    {
        double sum = 0;

        for (int j = 0; j < kernelN; j++)
            sum += h_Data[i ^ j] * h_Kernel[j];

        h_Result[i] = (float)sum;
    }
}