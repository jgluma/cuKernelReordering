/**
 * @file Needle.cu
 * @details This file describes the functions belonging to Needle class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "Needle.h"
#include "Needle_kernel.cu"

Needle::Needle(int cols, int rows, int p)
{
	
	max_rows_input = cols;
    max_cols_input = cols;
    penalty        = p;

    max_rows       = max_rows_input + 1;
    max_cols       = max_cols_input + 1;

    size = max_cols * max_rows;

	
}


Needle::~Needle()
{
	//Free host memory
    if(referrence      != NULL) cudaFreeHost(referrence);      
    if(input_itemsets  != NULL) cudaFreeHost(input_itemsets);  
    if(output_itemsets != NULL) cudaFreeHost(output_itemsets); 
    
    //Free device memory
    if(referrence_cuda != NULL) cudaFree(referrence_cuda);
    if(matrix_cuda     != NULL) cudaFree(matrix_cuda);
}

void Needle::allocHostMemory(void)
{
	cudaMallocHost((void **)&referrence,      max_rows * max_cols * sizeof(int));
    cudaMallocHost((void **)&input_itemsets,  max_rows * max_cols * sizeof(int));
    cudaMallocHost((void **)&output_itemsets, max_rows * max_cols * sizeof(int));
}

void Needle::freeHostMemory(void)
{
	if(referrence      != NULL) cudaFreeHost(referrence);      
    if(input_itemsets  != NULL) cudaFreeHost(input_itemsets);  
    if(output_itemsets != NULL) cudaFreeHost(output_itemsets); 
}

void Needle::allocDeviceMemory(void)
{
	cudaMalloc((void**)& referrence_cuda, sizeof(int)*size);
    cudaMalloc((void**)& matrix_cuda,     sizeof(int)*size);	
}

void Needle::freeDeviceMemory(void)
{
	if(referrence_cuda != NULL) cudaFree(referrence_cuda);
    if(matrix_cuda     != NULL) cudaFree(matrix_cuda);
}

void Needle::generatingData(void)
{
	srand ( 7 );
    
    
    for (int i = 0 ; i < max_cols; i++){
        for (int j = 0 ; j < max_rows; j++){
            input_itemsets[i*max_cols+j] = 0;
        }
    }

    for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
       input_itemsets[i*max_cols] = rand() % 10 + 1;
    }
    for( int j=1; j< max_cols ; j++){    //please define your own sequence.
       input_itemsets[j] = rand() % 10 + 1;
    }


    for (int i = 1 ; i < max_cols; i++){
        for (int j = 1 ; j < max_rows; j++){
        referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
        }
    }

    for( int i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
    for( int j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;
	
}

void Needle::memHostToDeviceAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(referrence_cuda, referrence,     sizeof(int) * size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(matrix_cuda,     input_itemsets, sizeof(int) * size, cudaMemcpyHostToDevice, stream);
}

void Needle::memHostToDevice(void)
{
    cudaMemcpy(referrence_cuda, referrence,     sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_cuda,     input_itemsets, sizeof(int) * size, cudaMemcpyHostToDevice);
}

void Needle::memDeviceToHostAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(output_itemsets, matrix_cuda, sizeof(int) * size, cudaMemcpyDeviceToHost, stream);
}

void Needle::memDeviceToHost(void)
{
   cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size, cudaMemcpyDeviceToHost);
}

void Needle::launch_kernel_Async(cudaStream_t stream)
{
	dim3 dimGrid;
    dim3 dimBlock(BLOCK_SIZE, 1);
    int block_width = ( max_cols - 1 )/BLOCK_SIZE;

    for( int i = 1 ; i <= block_width ; i++){
        dimGrid.x = i;
        dimGrid.y = 1;
        needle_cuda_shared_1<<<dimGrid, dimBlock, 0, stream>>>(referrence_cuda, matrix_cuda,
            max_cols, penalty, i, block_width); 
    }

    //process bottom-right matrix
    for( int i = block_width - 1  ; i >= 1 ; i--){
        dimGrid.x = i;
        dimGrid.y = 1;
        needle_cuda_shared_2<<<dimGrid, dimBlock, 0, stream>>>(referrence_cuda, matrix_cuda,
            max_cols, penalty, i, block_width); 
    }
	
	
	
}

void Needle::launch_kernel(void)
{
	dim3 dimGrid;
    dim3 dimBlock(BLOCK_SIZE, 1);
    int block_width = ( max_cols - 1 )/BLOCK_SIZE;

    for( int i = 1 ; i <= block_width ; i++){
        dimGrid.x = i;
        dimGrid.y = 1;
        needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence_cuda, matrix_cuda,
            max_cols, penalty, i, block_width); 
    }

    //process bottom-right matrix
    for( int i = block_width - 1  ; i >= 1 ; i--){
        dimGrid.x = i;
        dimGrid.y = 1;
        needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence_cuda, matrix_cuda,
            max_cols, penalty, i, block_width); 
    }
        
}

void Needle::checkResults(void)
{
	
}

void Needle::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = 2*(sizeof(int) * size);
	
	
}

void Needle::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = sizeof(int) * size;
	
	
}

void Needle::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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

