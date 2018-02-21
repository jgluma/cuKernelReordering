/**
 * @file transpose.cu
 * @details This file describes the functions belonging to Transpose class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "transpose.h"
#include "transpose_kernel.cu"


Transpose::Transpose(int w, int h)
{
	
	width       = w;
	height      = h;
	size_matrix = width * height;

}

Transpose::~Transpose()
{
	//Free host memory
	if(h_idata_TM    != NULL)      cudaFreeHost(h_idata_TM);
	if(h_odata_TM    != NULL)      cudaFreeHost(h_odata_TM);
	if(transposeGold != NULL)      delete [] transposeGold;

	//Free device memory
	if(d_idata_TM    != NULL)	   cudaFree(d_idata_TM);
	if(d_odata_TM    != NULL)	   cudaFree(d_odata_TM);	

}

void Transpose::allocHostMemory(void)
{
		
	
	cudaMallocHost((void **)&h_idata_TM, size_matrix * sizeof(float));
	cudaMallocHost((void **)&h_odata_TM,  size_matrix * sizeof(float));
    transposeGold = new float[size_matrix];

   
	
	
}

void Transpose::freeHostMemory(void)
{
	
	if(h_idata_TM    != NULL)      cudaFreeHost(h_idata_TM);
	if(h_odata_TM    != NULL)      cudaFreeHost(h_odata_TM);
	if(transposeGold != NULL)      delete [] transposeGold;
	
	
}

void Transpose::allocDeviceMemory(void)
{
	
	cudaMalloc((void **) &d_idata_TM, size_matrix * sizeof(float));
    cudaMalloc((void **) &d_odata_TM, size_matrix * sizeof(float));
	
}

void Transpose::freeDeviceMemory(void)
{
	if(d_idata_TM != NULL)	cudaFree(d_idata_TM);
	if(d_odata_TM != NULL)	cudaFree(d_odata_TM);
}

void Transpose::generatingData(void)
{
	for(int i = 0; i < size_matrix; i++)
			 h_idata_TM[i] = (float) i;
	
	
	
}

void Transpose::memHostToDeviceAsync(cudaStream_t stream)
{
	
	cudaMemcpyAsync(d_idata_TM, h_idata_TM, size_matrix*sizeof(float), cudaMemcpyHostToDevice, stream);
	
	
}

void Transpose::memHostToDevice(void)
{

    cudaMemcpy(d_idata_TM, h_idata_TM, size_matrix*sizeof(float), cudaMemcpyHostToDevice);


}

void Transpose::memDeviceToHostAsync(cudaStream_t stream)
{
	
	cudaMemcpyAsync(h_odata_TM, d_odata_TM, size_matrix*sizeof(float), cudaMemcpyDeviceToHost, stream);
	
}

void Transpose::memDeviceToHost(void)
{

    cudaMemcpy(h_odata_TM, d_odata_TM, size_matrix*sizeof(float), cudaMemcpyDeviceToHost);



}

void Transpose::launch_kernel_Async(cudaStream_t stream)
{
		
	dim3 grid(width/TILE_DIM, height/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);

	transposeNoBankConflicts<<<grid, threads, 0, stream>>>(d_odata_TM, d_idata_TM, width, height, 1);
	
	
}

void Transpose::launch_kernel(void)
{
	dim3 grid(width/TILE_DIM, height/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);

	transposeNoBankConflicts<<<grid, threads>>>(d_odata_TM, d_idata_TM, width, height, 1);
        


}

void Transpose::computeTransposeGold(float *gold, float *idata,
                          const  int w, const  int h)
{
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            gold[(x * h) + y] = idata[(y * w) + x];
        }
    }
}

void Transpose::checkResults(void)
{
	
	computeTransposeGold(transposeGold, h_idata_TM, width, height);
	
	bool success = true;
	
	for(int y = 0; y < height; y++)
	{
		for(int x = 0; x < width; x++)
			if(transposeGold[(y*width + x)] != h_odata_TM[(y*width + x)])
				success = false;
		
		
	}
	
	if(!success)
		printf("Error Transpose Matrix\n");
	
	
}


void Transpose::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = size_matrix * sizeof(float);
	
	
}

void Transpose::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = size_matrix * sizeof(float);
	
	
}

void Transpose::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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
