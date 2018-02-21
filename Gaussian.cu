/**
 * @file Gaussian.cu
 * @details This file describes the functions belonging to Gaussian class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "Gaussian.h"
#include "Gaussian_kernel.cu"

Gaussian::Gaussian(int s)
{
	Size = s;
}


Gaussian::~Gaussian()
{
	
    //Free host memory
    if(a        != NULL) cudaFreeHost(a);
    if(b        != NULL) cudaFreeHost(b);
    if(m        != NULL) cudaFreeHost(m);
    if(finalVec != NULL) delete [] finalVec;
    
    //Free device memory
    if(m_cuda   != NULL) cudaFree(m_cuda);
    if(a_cuda   != NULL) cudaFree(a_cuda);
    if(b_cuda   != NULL) cudaFree(b_cuda);
}

void Gaussian::allocHostMemory(void)
{
	cudaMallocHost((void **)&a, Size * Size * sizeof(float));
    cudaMallocHost((void **)&b, Size * Size * sizeof(float));
    cudaMallocHost((void **)&m, Size * Size * sizeof(float));
    finalVec = new float[Size];
}

void Gaussian::freeHostMemory(void)
{
	//Free host memory
    if(a        != NULL) cudaFreeHost(a);
    if(b        != NULL) cudaFreeHost(b);
    if(m        != NULL) cudaFreeHost(m);
    if(finalVec != NULL) delete [] finalVec;
}

void Gaussian::allocDeviceMemory(void)
{
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
    cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
    cudaMalloc((void **) &b_cuda, Size * sizeof(float));	
}

void Gaussian::freeDeviceMemory(void)
{
	if(m_cuda != NULL) cudaFree(m_cuda);
    if(a_cuda != NULL) cudaFree(a_cuda);
    if(b_cuda != NULL) cudaFree(b_cuda);
}

void Gaussian::generatingData(void)
{
	create_matrix(a, Size);

    for (int j =0; j< Size; j++)
            b[j]=1.0;

    for (int i=0; i<Size*Size; i++)
            m[i] = 0.0;
	
}

void Gaussian::memHostToDeviceAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(m_cuda, m, Size * Size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(a_cuda, a, Size * Size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b_cuda, b, Size * sizeof(float),        cudaMemcpyHostToDevice, stream);
}

void Gaussian::memHostToDevice(void)
{
    cudaMemcpy(m_cuda, m, Size * Size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_cuda, a, Size * Size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, Size * sizeof(float),        cudaMemcpyHostToDevice);
}

void Gaussian::memDeviceToHostAsync(cudaStream_t stream)
{
	cudaMemcpyAsync(m, m_cuda, Size * Size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(a, a_cuda, Size * Size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(b, b_cuda, Size * sizeof(float),        cudaMemcpyDeviceToHost, stream);
}

void Gaussian::memDeviceToHost(void)
{
    cudaMemcpy(m, m_cuda, Size * Size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(a, a_cuda, Size * Size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, b_cuda, Size * sizeof(float),        cudaMemcpyDeviceToHost);
}

void Gaussian::launch_kernel_Async(cudaStream_t stream)
{
	int block_size,grid_size;
    
    block_size = MAXBLOCKSIZE_GAUSSIAN;
    grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);

    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);

    int blockSize2d, gridSize2d;
    blockSize2d = BLOCK_SIZE_XY_GAUSSIAN;
    gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 
    
    dim3 dimBlockXY(blockSize2d,blockSize2d);
    dim3 dimGridXY(gridSize2d,gridSize2d);

    for (int t=0; t<(Size-1); t++) {
        Fan1<<<dimGrid,dimBlock, 0, stream>>>(m_cuda,a_cuda,Size,t);
        Fan2<<<dimGridXY,dimBlockXY, 0, stream>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
        
    }
	
	
	
}

void Gaussian::launch_kernel(void)
{
	int block_size,grid_size;
    
    block_size = MAXBLOCKSIZE_GAUSSIAN;
    grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);

    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);

    int blockSize2d, gridSize2d;
    blockSize2d = BLOCK_SIZE_XY_GAUSSIAN;
    gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 
    
    dim3 dimBlockXY(blockSize2d,blockSize2d);
    dim3 dimGridXY(gridSize2d,gridSize2d);

    for (int t=0; t<(Size-1); t++) {
        Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
        Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
        
    }
        
}

void Gaussian::checkResults(void)
{
	BackSub();
}

void Gaussian::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = (2*(Size * Size * sizeof(float))) + (Size * sizeof(float));
	
	
}

void Gaussian::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = (2*(Size * Size * sizeof(float))) + (Size * sizeof(float));
	
	
}

void Gaussian::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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

void Gaussian::create_matrix(float *m, int size)
{
    int i,j;
    float lamda = -0.01;
    float coe[2*size-1];
    float coe_i =0.0;

    for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }


    for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
        m[i*size+j]=coe[size-1-i+j];
      }
    }
}

void Gaussian::BackSub(void)
{
    // solve "bottom up"
    int i,j;
    for(i=0;i<Size;i++){
        finalVec[Size-i-1]=b[Size-i-1];
        for(j=0;j<i;j++)
        {
            finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
        }
        finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
    }
}


