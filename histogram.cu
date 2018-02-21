/**
 * @file histogram.cu
 * @details This file describes the functions belonging to Histogram class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "histogram.h"
#include "histogram_kernel.cu"


Histogram::Histogram(int nframes)
{
	//Number of frames
	frames = nframes;
	
	DATA_W = iAlignUp(704, 16);
	DATA_H = 576;
	DATA_SIZE = DATA_W * DATA_H;
	DATA_SIZE4 = DATA_W * DATA_H / 4;
	DATA_SIZE_INT = DATA_SIZE4 * sizeof(unsigned int);
	
	

}

int Histogram::iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

Histogram::~Histogram()
{
	//Free host memory
	if(h_DataA!=NULL)        	cudaFreeHost(h_DataA);
	if(h_histoCPU!=NULL)        free(h_histoCPU);
	if(h_histoGPU!=NULL)        cudaFreeHost(h_histoGPU);
	if(hh_DataA!=NULL)	        cudaFreeHost(hh_DataA);

	//Free device memory
	if(d_DataA!=NULL)	cudaFree(d_DataA);
	if(d_histo!=NULL)	cudaFree(d_histo);
	if(d_zero!=NULL)	cudaFree(d_zero);

}

void Histogram::allocHostMemory(void)
{
		
	
	cudaMallocHost((void **)&h_DataA, frames * DATA_SIZE_INT);
	h_histoCPU = (unsigned int *)malloc(frames * HISTOGRAM_SIZE);
	cudaMallocHost((void **)&h_histoGPU, frames * HISTOGRAM_SIZE);
	cudaMallocHost((void **)&hh_DataA, frames * DATA_SIZE*sizeof(unsigned char));
	
	
}

void Histogram::freeHostMemory(void)
{
	
	//Free host memory
	if(h_DataA!=NULL)        	cudaFreeHost(h_DataA);
	if(h_histoCPU!=NULL)        free(h_histoCPU);
	if(h_histoGPU!=NULL)        cudaFreeHost(h_histoGPU);
	if(hh_DataA!=NULL)	        cudaFreeHost(hh_DataA);
	
}

void Histogram::allocDeviceMemory(void)
{
	
	cudaMalloc((void **)&d_DataA, frames * DATA_SIZE_INT);
	cudaMalloc((void **)&d_histo, frames * HISTOGRAM_SIZE);
	cudaMalloc((void **)&d_zero, frames * HISTOGRAM_SIZE);
	cudaMemset(d_zero, 0, frames * HISTOGRAM_SIZE);
	
	
}

void Histogram::freeDeviceMemory(void)
{
	
	//Free device memory
	if(d_DataA!=NULL)	cudaFree(d_DataA);
	if(d_histo!=NULL)	cudaFree(d_histo);
	if(d_zero!=NULL)	cudaFree(d_zero);	
}

void Histogram::generatingData(void)
{
	
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Input images generation
	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	srand(time(NULL));
	
	for(int i = 0; i < frames*DATA_SIZE4; i++)
		hh_DataA[i] = (unsigned char)(rand() % 256);
	
		
	memcpy(h_DataA, hh_DataA, DATA_SIZE_INT*frames);
	
	
}

void Histogram::memHostToDeviceAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(d_DataA, 
					 h_DataA, 
					 DATA_SIZE_INT*frames, 
					 cudaMemcpyHostToDevice, stream);
	
}

void Histogram::memHostToDevice(void)
{

    cudaMemcpy(d_DataA, h_DataA, DATA_SIZE_INT*frames, cudaMemcpyHostToDevice);


}

void Histogram::memDeviceToHostAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(h_histoGPU, 
					d_histo, 
					frames*HISTOGRAM_SIZE,cudaMemcpyDeviceToHost, stream);
	
}

void Histogram::memDeviceToHost(void)
{

        cudaMemcpy(h_histoGPU, 
				   d_histo, 
				   frames*HISTOGRAM_SIZE,cudaMemcpyDeviceToHost);



}

void Histogram::launch_kernel_Async(cudaStream_t stream)
{
	histogram256Kernel<<<DATA_SIZE4*frames/THREAD_N, THREAD_N, 0, stream>>>(d_histo, 
																			(unsigned int *)d_DataA, 
																			frames*DATA_SIZE/4);
	
	
	
}

void Histogram::launch_kernel(void)
{
	histogram256Kernel<<<DATA_SIZE4*frames/THREAD_N, THREAD_N>>>(d_histo, 
																(unsigned int *)d_DataA, 
																frames*DATA_SIZE/4);
	
        


}

void Histogram::histogram256CPU(
    unsigned int *h_Result,
    unsigned int *h_Data,
    int dataN
){
    int i;
    unsigned int data4;

    for (i = 0; i < dataN; i++){
        data4 = h_Data[i];
        h_Result[(data4 >>  0) & 0xFF]++;
        h_Result[(data4 >>  8) & 0xFF]++;
        h_Result[(data4 >> 16) & 0xFF]++;
        h_Result[(data4 >> 24) & 0xFF]++;
    }
}

void Histogram::checkResults(void)
{
	
	double sum_delta2,sum_ref2,L1norm2;
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// CPU computing
	///////////////////////////////////////////////////////////////////////////////////////////////////
		//memset(h_histoCPU + idx_chunk * max_framesPerChunk_HST, 0, HISTOGRAM_SIZE*framesPerChunk_HST[idx_chunk]);
		memset(h_histoCPU, 0, HISTOGRAM_SIZE);
		
		//for(int i = 0; i < FRAMES; ++i){
			//histogram256CPU(h_histoCPU+i*BIN_COUNT, (unsigned int *)h_DataA+i*DATA_SIZE4, DATA_SIZE/4);
		//}
		
		for(int i = 0; i < frames; ++i){
			histogram256CPU(h_histoCPU + i*BIN_COUNT, 
							(unsigned int *)h_DataA + i*DATA_SIZE4, 
							DATA_SIZE/4);
		}
		
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Comparing the results
	///////////////////////////////////////////////////////////////////////////////////////////////////
       

	sum_delta2 = 0;
    sum_ref2   = 0;
    L1norm2 = 0;
    for(int i = 0; i < BIN_COUNT*frames; i++){
            sum_delta2 += fabs(h_histoCPU[i] - h_histoGPU[i]);
            sum_ref2   += fabs(h_histoCPU[i]);
    }

    L1norm2 = sum_delta2 / sum_ref2;
    if(L1norm2 >= 1e-6)
		printf("HST FAILED!\n");
	
	
}


void Histogram::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = DATA_SIZE_INT*frames;
	
	
}

void Histogram::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = frames*HISTOGRAM_SIZE;
	
	
}

void Histogram::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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
