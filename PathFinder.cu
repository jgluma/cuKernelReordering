/**
 * @file PathFinder.cu
 * @details This file describes the functions belonging to PathFinder class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "PathFinder.h"
#include "PathFinder_kernel.cu"

PathFinder::PathFinder(int c, int r, int p)
{
	cols = c;
    rows = r;
    pyramid_height = p;

    /* --------------- pyramid parameters --------------- */
    borderCols = (pyramid_height)*HALO_PATH_FINDER;
    smallBlockCol = BLOCK_SIZE_PATH_FINDER-(pyramid_height)*HALO_PATH_FINDER*2;
    blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    size = rows*cols;

    seed = M_SEED_PATH_FINDER;
	

	
}


PathFinder::~PathFinder()
{
	//Free host memory
    if(data != NULL) cudaFreeHost(data);
    if(wall != NULL) delete [] wall;

    //Device memory
    if(gpuResult[0] != NULL)      cudaFree(gpuResult[0]);
    if(gpuResult[1] != NULL)      cudaFree(gpuResult[1]);
    if(gpuWall      != NULL)      cudaFree(gpuWall);
}

void PathFinder::allocHostMemory(void)
{
	cudaMallocHost((void **)&data, rows*cols * sizeof(int));

    wall = new int*[rows];

    cudaMallocHost((void **)&result, cols*sizeof(int));
}

void PathFinder::freeHostMemory(void)
{
	if(data != NULL) cudaFreeHost(data);
    if(wall != NULL) delete [] wall;
    if(result != NULL) cudaFreeHost(result);
}

void PathFinder::allocDeviceMemory(void)
{
	cudaMalloc((void**)&gpuResult[0], sizeof(int)*cols);
    cudaMalloc((void**)&gpuResult[1], sizeof(int)*cols);
    cudaMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
}

void PathFinder::freeDeviceMemory(void)
{
	if(gpuResult[0] != NULL)      cudaFree(gpuResult[0]);
    if(gpuResult[1] != NULL)      cudaFree(gpuResult[1]);
    if(gpuWall      != NULL)      cudaFree(gpuWall);
}

void PathFinder::generatingData(void)
{
	for(int n=0; n<rows; n++)
        wall[n]=data+cols*n;

    srand(seed);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            wall[i][j] = rand() % 10;

}

void PathFinder::memHostToDeviceAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(gpuResult[0], data,      sizeof(int)*cols,        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpuWall,      data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice, stream);
}

void PathFinder::memHostToDevice(void)
{
    cudaMemcpy(gpuResult[0], data,      sizeof(int)*cols,        cudaMemcpyHostToDevice);
    cudaMemcpy(gpuWall,      data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);
}

void PathFinder::memDeviceToHostAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost, stream);
}

void PathFinder::memDeviceToHost(void)
{
   cudaMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);
}

void PathFinder::launch_kernel_Async(cudaStream_t stream)
{
	dim3 dimBlock(BLOCK_SIZE_PATH_FINDER);
    dim3 dimGrid(blockCols);  
    
    int src = 1, dst = 0;

    
            int temp = src;
            src = dst;
            dst = temp;
            dynproc_kernel<<<dimGrid, dimBlock, 0, stream>>>( 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, borderCols, pyramid_height);
    

    final_ret = dst;
	
}

void PathFinder::launch_kernel(void)
{
	dim3 dimBlock(BLOCK_SIZE_PATH_FINDER);
    dim3 dimGrid(blockCols);  
    
    int src = 1, dst = 0;

    
            int temp = src;
            src = dst;
            dst = temp;
            dynproc_kernel<<<dimGrid, dimBlock>>>( 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, borderCols, pyramid_height);
    

    final_ret = dst;
        
}

void PathFinder::checkResults(void)
{
	
}

void PathFinder::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = (cols + (size-cols))*sizeof(int);
	
	
}

void PathFinder::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = sizeof(int)*cols;
	
	
}

void PathFinder::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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

