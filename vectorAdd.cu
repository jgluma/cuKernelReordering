/**
 * @file vectorAdd.cu
 * @details This file describes the functions belonging to VectorADD class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "vectorAdd.h"
#include "vectorAdd_kernel.cu"

VectorADD::VectorADD()
{
	h_A_VA = NULL;
	h_B_VA = NULL;
	h_C_VA = NULL;
	d_A_VA = NULL;
	d_B_VA = NULL;
	d_C_VA = NULL;

	n_vector_VA        = 0;
	numElements_VA     = NULL;
	max_numElements_VA = 0;
}

VectorADD::VectorADD(int size)
{
	h_A_VA = NULL;
	h_B_VA = NULL;
	h_C_VA = NULL;
	d_A_VA = NULL;
	d_B_VA = NULL;
	d_C_VA = NULL;

	n_vector_VA        = 0;
	numElements_VA     = NULL;
	max_numElements_VA = 0;

	n_vector_VA = 1;
	

	numElements_VA = new int [n_vector_VA];
	
	for(int i = 0; i < n_vector_VA; i++)
		numElements_VA[i] = size;
	

	max_numElements_VA = 0;
	
	for(int i = 0; i < n_vector_VA; i++)
	{
		if(max_numElements_VA < numElements_VA[i])
			max_numElements_VA = numElements_VA[i];
		
	}

}

VectorADD::~VectorADD()
{
	//Free host memory
	if(h_A_VA!=NULL)        	cudaFreeHost(h_A_VA);
	if(h_B_VA!=NULL)        	cudaFreeHost(h_B_VA);
	if(h_C_VA!=NULL)        	cudaFreeHost(h_C_VA);
	if(numElements_VA!=NULL)	delete [] numElements_VA;

	//Free device memory
	if(d_A_VA!=NULL)	cudaFree(d_A_VA);
	if(d_B_VA!=NULL)	cudaFree(d_B_VA);
	if(d_C_VA!=NULL)	cudaFree(d_C_VA);	

}

void VectorADD::allocHostMemory(void)
{
		
	
	cudaMallocHost((void **)&h_A_VA, n_vector_VA * max_numElements_VA * sizeof(float));
	cudaMallocHost((void **)&h_B_VA, n_vector_VA * max_numElements_VA * sizeof(float));
	cudaMallocHost((void **)&h_C_VA, n_vector_VA * max_numElements_VA * sizeof(float));
	
	
}

void VectorADD::freeHostMemory(void)
{
	
	if(h_A_VA!=NULL)        	cudaFreeHost(h_A_VA);
	if(h_B_VA!=NULL)        	cudaFreeHost(h_B_VA);
	if(h_C_VA!=NULL)        	cudaFreeHost(h_C_VA);
	if(numElements_VA!=NULL)	delete [] numElements_VA;
	
}

void VectorADD::allocDeviceMemory(void)
{
	
	cudaMalloc((void **)&d_A_VA, n_vector_VA * max_numElements_VA * sizeof(float));
	cudaMalloc((void **)&d_B_VA, n_vector_VA * max_numElements_VA * sizeof(float));
	cudaMalloc((void **)&d_C_VA, n_vector_VA * max_numElements_VA * sizeof(float));
	
	
}

void VectorADD::freeDeviceMemory(void)
{
	
	if(d_A_VA!=NULL)	cudaFree(d_A_VA);
	if(d_B_VA!=NULL)	cudaFree(d_B_VA);
	if(d_C_VA!=NULL)	cudaFree(d_C_VA);	
}

void VectorADD::generatingData(void)
{
	
	for (int i = 0; i < n_vector_VA; i++)
	{
			
			for(int j = 0; j < numElements_VA[i]; j++)
			{
				h_A_VA[i*max_numElements_VA + j] = rand()/(float)RAND_MAX;
				h_B_VA[i*max_numElements_VA + j] = rand()/(float)RAND_MAX;
			}
	}
	
}

void VectorADD::memHostToDeviceAsync(cudaStream_t stream)
{
	
	int idx_vector = 0;
	cudaMemcpyAsync(d_A_VA + idx_vector * max_numElements_VA, h_A_VA + idx_vector * max_numElements_VA, numElements_VA[idx_vector]*sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_B_VA + idx_vector * max_numElements_VA, h_B_VA + idx_vector * max_numElements_VA, numElements_VA[idx_vector]*sizeof(float), cudaMemcpyHostToDevice, stream);
	
	
}

void VectorADD::memHostToDevice(void)
{

        int idx_vector = 0;
        cudaMemcpy(d_A_VA + idx_vector * max_numElements_VA, 
		h_A_VA + idx_vector * max_numElements_VA, 
		numElements_VA[idx_vector]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_VA + idx_vector * max_numElements_VA, 
		h_B_VA + idx_vector * max_numElements_VA, 
		numElements_VA[idx_vector]*sizeof(float), cudaMemcpyHostToDevice);


}

void VectorADD::memDeviceToHostAsync(cudaStream_t stream)
{
	
	int idx_vector = 0;
	cudaMemcpyAsync(h_C_VA + idx_vector * max_numElements_VA, d_C_VA + idx_vector * max_numElements_VA, numElements_VA[idx_vector]*sizeof(float), cudaMemcpyDeviceToHost, stream);
	
	
	
}

void VectorADD::memDeviceToHost(void)
{

        int idx_vector = 0;
        cudaMemcpy(h_C_VA + idx_vector * max_numElements_VA, 
		d_C_VA + idx_vector * max_numElements_VA, 
		numElements_VA[idx_vector]*sizeof(float), cudaMemcpyDeviceToHost);



}

void VectorADD::launch_kernel_Async(cudaStream_t stream)
{
	
	int idx_vector = 0;
	int threadsPerBlock = 256;
	int blocksPerGrid = (ceil((float)numElements_VA[idx_vector]/(float)threadsPerBlock));
	
	vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A_VA + idx_vector * max_numElements_VA, d_B_VA + idx_vector * max_numElements_VA, d_C_VA + idx_vector * max_numElements_VA, numElements_VA[idx_vector], 0);
	
	
}

void VectorADD::launch_kernel(void)
{

        int idx_vector = 0;
        int threadsPerBlock = 256;
        int blocksPerGrid = (ceil((float)numElements_VA[idx_vector]/(float)threadsPerBlock));

        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A_VA + idx_vector * max_numElements_VA, d_B_VA + idx_vector * max_numElements_VA, d_C_VA + idx_vector * max_numElements_VA, numElements_VA[idx_vector], 0);


}

void VectorADD::checkResults(void)
{
	
	int idx_vector = 0;

	for (int i = 0; i < numElements_VA[idx_vector]; ++i)
	{
			
			//printf("i: %d - A: %f - B: %f - C: %f\n", i, h_A_vectorAdd[i], h_B_vectorAdd[i], h_C_vectorAdd[i]);
		   if (fabs(h_A_VA[idx_vector * max_numElements_VA + i] + h_B_VA[idx_vector * max_numElements_VA + i] - h_C_VA[idx_vector * max_numElements_VA + i]) > 1e-5)
			{
				printf("Result verification failed at element %d!\n", i);
				
			}
	}
	
	
}


void VectorADD::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = 2*(numElements_VA[0]*sizeof(float));
	
	
}

void VectorADD::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = numElements_VA[0]*sizeof(float);
	
	
}

void VectorADD::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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
