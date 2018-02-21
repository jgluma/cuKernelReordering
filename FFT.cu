/**
 * @file FFT.cu
 * @details This file describes the functions belonging to FFT class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "FFT.h"


FFT::FFT(int s)
{
	size = s;	

}

FFT::~FFT()
{
	//Free host memory
	if(h_in  !=NULL)    cudaFreeHost(h_in);
	if(h_out !=NULL)    cudaFreeHost(h_out);

	//Free device memory
	if(d_in  != NULL)	cudaFree(d_in);
	if(d_out != NULL)	cudaFree(d_out);	

}

void FFT::allocHostMemory(void)
{
		
	
	cudaMallocHost((void **)&h_in, size*sizeof(float2));
    cudaMallocHost((void **)&h_out, size*sizeof(float2));
	
	
}

void FFT::freeHostMemory(void)
{
	
	if(h_in!=NULL)        	cudaFreeHost(h_in);
	if(h_out!=NULL)        	cudaFreeHost(h_out);
	
	
}

void FFT::allocDeviceMemory(void)
{
	
	cudaMalloc((void **)&d_in,  size * sizeof(float2));
	cudaMalloc((void **)&d_out, size * sizeof(float2));
	
	
}

void FFT::freeDeviceMemory(void)
{
	
	if(d_in  != NULL)	cudaFree(d_in);
	if(d_out != NULL)	cudaFree(d_out);
	
}

void FFT::generatingData(void)
{
	
	//Generating input data
	for (int i = 0; i < size; i++) {
        h_in[i].x = 1.f;
        h_in[i].y = 0.f;
        
    }
   
    for (int i = 0; i < size; i++) {
        h_out[i].x = 0.f;
        h_out[i].y = 0.f;
       
    }

    //Creating FFT plan
    cufftPlan1d(&plan, size, CUFFT_C2C, 1);
	
}

void FFT::memHostToDeviceAsync(cudaStream_t stream)
{
	
	cudaMemcpyAsync(d_in, h_in, size*sizeof(float2), cudaMemcpyHostToDevice, stream);
	
	
}

void FFT::memHostToDevice(void)
{

    cudaMemcpy(d_in, h_in, size*sizeof(float2), cudaMemcpyHostToDevice);


}

void FFT::memDeviceToHostAsync(cudaStream_t stream)
{
	
	cudaMemcpyAsync(h_out, d_out, size*sizeof(float2), cudaMemcpyDeviceToHost, stream);
	
	
	
}

void FFT::memDeviceToHost(void)
{

    cudaMemcpy(h_out, d_out, size*sizeof(float2), cudaMemcpyDeviceToHost);



}

void FFT::launch_kernel_Async(cudaStream_t stream)
{
	
	

	cufftSetStream(plan, stream);
    
    cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_FORWARD);
	
	
}

void FFT::launch_kernel(void)
{

    cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_FORWARD);  


}

void FFT::checkResults(void)
{
	
	
	
	
}


void FFT::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = size*sizeof(float2);
	
	
}

void FFT::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = size*sizeof(float2);
	
	
}

void FFT::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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
