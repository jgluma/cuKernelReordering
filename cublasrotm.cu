/**
 * @file cublasrotm.cu
 * @details This file describes the functions belonging to CUBLASROTM class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "cublasrotm.h"


CUBLASROTM::CUBLASROTM(int s)
{
	h_x = NULL;
	h_y = NULL;
	d_x = NULL;
	d_y = NULL;
	h_H = NULL;
	d_H = NULL;

	size = s;

}

CUBLASROTM::~CUBLASROTM()
{
	//Free host memory
	if(h_x != NULL)   cudaFreeHost(h_x);
	if(h_y != NULL)   cudaFreeHost(h_y);
	if(h_H != NULL)   cudaFreeHost(h_H);

	//Free device memory
	if(d_x != NULL)	cudaFree(d_x);
	if(d_y != NULL)	cudaFree(d_y);	

}

void CUBLASROTM::allocHostMemory(void)
{
		
	
	cudaMallocHost((void **)&h_x, size * sizeof(float));
	cudaMallocHost((void **)&h_y, size * sizeof(float));
	cudaMallocHost((void **)&h_H, 5 * sizeof(float));
	
	
}

void CUBLASROTM::freeHostMemory(void)
{
	
	if(h_x != NULL)   cudaFreeHost(h_x);
	if(h_y != NULL)   cudaFreeHost(h_y);
	if(h_H != NULL)   cudaFreeHost(h_H);
	
	
}

void CUBLASROTM::allocDeviceMemory(void)
{
	
	cudaMalloc((void **)&d_x, size * sizeof(float));
	cudaMalloc((void **)&d_y, size * sizeof(float));
	
	
}

void CUBLASROTM::freeDeviceMemory(void)
{
	
	if(d_x != NULL)	cudaFree(d_x);
	if(d_y != NULL)	cudaFree(d_y);
	
}

void CUBLASROTM::generatingData(void)
{
	
	//Generating Data vectors
	srand(time(NULL));
	
	for(int i = 0; i < size; i++)
	{
		h_x[i] = (float)(1 + rand()%1000);
		h_y[i] = (float)(1 + rand()%1000);
	}
		
		
	
	
	
	h_H[0] = -1.0;
	for(int i = 0; i < 4; i++)
	{
		h_H[1 + i] = (float)(1 + rand()%40);
		
	}
	
	
	//Creating handle CUBLAS
	cublasCreate(&handle);
	
}

void CUBLASROTM::memHostToDeviceAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(d_x, h_x, sizeof(float)*size, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_y, h_y, sizeof(float)*size, cudaMemcpyHostToDevice, stream);
}

void CUBLASROTM::memHostToDevice(void)
{
   	cudaMemcpy(d_x, h_x, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, sizeof(float)*size, cudaMemcpyHostToDevice);
}

void CUBLASROTM::memDeviceToHostAsync(cudaStream_t stream)
{
	//Load the column indices to the gpu
	cudaMemcpyAsync(h_x, d_x, sizeof(float)*size, cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(h_y, d_y, sizeof(float)*size, cudaMemcpyDeviceToHost, stream);
}

void CUBLASROTM::memDeviceToHost(void)
{

  //Load the column indices to the gpu
  cudaMemcpyAsync(h_x, d_x, sizeof(float)*size, cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_y, d_y, sizeof(float)*size, cudaMemcpyDeviceToHost);

}

void CUBLASROTM::launch_kernel_Async(cudaStream_t stream)
{
	
	cublasStatus_t status;
	 
	cublasSetStream(handle, stream);
	
	status = cublasSrotm(handle, size, d_x, incx, d_y, incy, h_H);
	if(status != CUBLAS_STATUS_SUCCESS )
	{
		cout << "Error CUBLAS" << endl;
		exit(1);
		
	}
	
	
}

void CUBLASROTM::launch_kernel(void)
{

    cublasStatus_t status;
	
	status = cublasSrotm(handle, size, d_x, incx, d_y, incy, h_H);
	if(status != CUBLAS_STATUS_SUCCESS )
	{
		cout << "Error CUBLAS" << endl;
		exit(1);
		
	}


}

void CUBLASROTM::checkResults(void)
{
	
	
}


void CUBLASROTM::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = (sizeof(float)*size)*2;
	
	
}

void CUBLASROTM::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = (sizeof(float)*size)*2;
	
	
}

void CUBLASROTM::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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
