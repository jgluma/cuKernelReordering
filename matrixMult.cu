/**
 * @file matrixMult.cu
 * @details This file describes the functions belonging to MatrixMult class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "matrixMult.h"
#include "matrixMult_kernel.cu"


MatrixMult::MatrixMult(int *sizes)
{
	valB = 0.01f;
			
	h_A_MM = NULL;
	h_B_MM = NULL;
	h_C_MM = NULL;
	d_A_MM = NULL;
	d_B_MM = NULL; 
	d_C_MM = NULL;

	uiWA_MM = sizes[0];
	uiHA_MM = sizes[1];
	uiWB_MM = sizes[2];
	uiHB_MM = sizes[3];
	uiWC_MM = uiWB_MM; 
	uiHC_MM = uiHA_MM;

	mem_size_A_MM = uiWA_MM * uiHA_MM;
	mem_size_B_MM = uiWB_MM * uiHB_MM;
	mem_size_C_MM = uiWC_MM * uiHC_MM;

	blockSize = 16;


}

MatrixMult::~MatrixMult()
{
	if(h_A_MM!=NULL)        	cudaFreeHost(h_A_MM);
	if(h_B_MM!=NULL)        	cudaFreeHost(h_B_MM);
	if(h_C_MM!=NULL)        	cudaFreeHost(h_C_MM);



}

void MatrixMult::allocHostMemory(void)
{
		
	
	cudaMallocHost((void **)&h_A_MM, mem_size_A_MM * sizeof(float));
	cudaMallocHost((void **)&h_B_MM, mem_size_B_MM * sizeof(float));
	cudaMallocHost((void **)&h_C_MM, mem_size_C_MM * sizeof(float));
	
	
}

void MatrixMult::freeHostMemory(void)
{
	
	if(h_A_MM!=NULL)        cudaFreeHost(h_A_MM);
	if(h_B_MM!=NULL)        cudaFreeHost(h_B_MM);
	if(h_C_MM!=NULL)        cudaFreeHost(h_C_MM);
	
	
}

void MatrixMult::allocDeviceMemory(void)
{
	
	cudaMalloc((void**) &d_A_MM, mem_size_A_MM * sizeof(float));
	cudaMalloc((void**) &d_B_MM, mem_size_B_MM * sizeof(float));
	cudaMalloc((void**) &d_C_MM, mem_size_C_MM * sizeof(float));
	
	
}

void MatrixMult::freeDeviceMemory(void)
{
	
	if(d_A_MM!=NULL)	cudaFree(d_A_MM);
	if(d_B_MM!=NULL)	cudaFree(d_B_MM);
	if(d_C_MM!=NULL)	cudaFree(d_C_MM);	
}

void MatrixMult::constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

void MatrixMult::generatingData(void)
{
	
	
	constantInit(h_A_MM, mem_size_A_MM, 1.0f);
	constantInit(h_B_MM, mem_size_B_MM, valB);
	
	
	cudaMemset(d_C_MM, 0, mem_size_C_MM * sizeof(float));
	
}

void MatrixMult::memHostToDeviceAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(d_A_MM, h_A_MM, mem_size_A_MM*sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_B_MM, h_B_MM, mem_size_B_MM*sizeof(float), cudaMemcpyHostToDevice, stream);
	
}

void MatrixMult::memHostToDevice(void)
{
	cudaMemcpy(d_A_MM, h_A_MM, mem_size_A_MM*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_MM, h_B_MM, mem_size_B_MM*sizeof(float), cudaMemcpyHostToDevice);
	
}

void MatrixMult::memDeviceToHostAsync(cudaStream_t stream)
{
	
	cudaMemcpyAsync(h_C_MM, d_C_MM, mem_size_C_MM*sizeof(float), cudaMemcpyDeviceToHost, stream);
	
	
	
}

void MatrixMult::memDeviceToHost(void)
{
	
	cudaMemcpy(h_C_MM, d_C_MM, mem_size_C_MM*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	
}

void MatrixMult::launch_kernel_Async(cudaStream_t stream)
{
	dim3 threadsS(blockSize, blockSize);
	dim3 blocksS(uiWC_MM/ threadsS.x, uiHC_MM / threadsS.y);
	
	matrixMul<<< blocksS, threadsS, 0, stream >>>(d_C_MM, 
												  d_A_MM, 
												  d_B_MM, uiWA_MM, uiWB_MM);
	
	
	
}

void MatrixMult::launch_kernel(void)
{
	dim3 threadsS(blockSize, blockSize);
	dim3 blocksS(uiWC_MM/ threadsS.x, uiHC_MM / threadsS.y);
	
	matrixMul<<< blocksS, threadsS>>>(d_C_MM, 
									  d_A_MM, 
									  d_B_MM, uiWA_MM, uiWB_MM);
	
}

void MatrixMult::checkResults(void)
{
	
	bool correct = true;

		//for (int i = 0; i < size_C_MM; i++)
		
	for (int i = 0; i < mem_size_C_MM; i++)
	{
			if (fabs(h_C_MM[i] - (uiWA_MM * valB)) > 0.1)//1e-5)
			{
				printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > 1e-5\n", i, h_C_MM[i], uiWA_MM*valB);
				correct = false;
			}
	}


	if(correct == false)
	{
			printf("Error Matrix Multiplication\n");
			
			
	}
	
	
}


void MatrixMult::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = mem_size_A_MM*sizeof(float) + mem_size_B_MM*sizeof(float);
	
	
}

void MatrixMult::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = mem_size_C_MM * sizeof(float);
	
	
}

void MatrixMult::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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