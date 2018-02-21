/**
 * @file microbenchmarking_transfers.cu
 * @detail This file describes the implementation of the functions involved in the
 * PCIe bandwidth and latency.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */

#include <stdio.h>

void swap(float & v1, float & v2){
    float tmp = v1;
    v1 = v2;
    v2 = tmp;
}


int partition(float *array, int left, int right){
    int part = right;
    swap(array[part],array[(right+left) / 2]);
    
    --right;
 
    while(true){
        while(array[left] < array[part]){
            ++left;
        }
        while(right >= left && array[part] <= array[right]){
            --right;
        }
        if(right < left) break;
 
        swap(array[left],array[right]);
        ++left;
        --right;
    }
 
    swap(array[part],array[left]);
 
    return left;
}

void qs(float * array, const int left, const int right){
    if(left < right){
        const int part = partition(array, left, right);
        qs(array, part + 1,right);
        qs(array, left,part - 1);
    }
}


/**
 * @brief Quicksort
 * @author Antonio Jose Lazaro Munoz
 * @date 17/02/2016
 * @details Quicksort
 * 
 * @param array elements array
 * @param size array size
 */
void serialQuickSort(float *array, const int size){
    qs(array, 0,size-1);
}

/**
 * @brief Calculate times median
 * @author Antonio Jose Lazaro Munoz
 * @date 17/02/2016
 * @details This function returns the median from a set of times.
 * 
 * @param h_times times array.
 * @param N array size.
 * 
 * @return time median.
 */
float getMedianTime(float *h_times, int N)
{
	float median = 0;

	float * h_sorted_times = (float *)malloc(N * sizeof(float));
	
	for(int n = 0; n < N; n++)
		h_sorted_times[n] = h_times[n];
		
	//Sort execution times
	serialQuickSort(h_sorted_times, N);
	
	//Calculate median
	if(N%2 == 0)
	{
		
		median = (h_sorted_times[N/2] + h_sorted_times[(N/2)+1])/2;
		
	}
	else
	{
		int p = N/2;
		
		median = h_sorted_times[p];
		
	}
	
	free(h_sorted_times);
	
	return median;
	
}

/**
 * @brief Calculate PCIe HTD latency.
 * @author Antonio Jose Lazaro Munoz
 * @date 17/02/2016
 * @details This function returns the latency of the PCIe for HTD memory transfers.
 * 
 * @param d_data device data.
 * @param h_data host data.
 * @param nreps iterations.
 * @return HTD PCIe latency (ms).
 */
float getLoHTD(char *d_data, char *h_data, int nreps)
{

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	float LoHTD = 0.0;
	
	
	float *h_times = (float *)malloc(nreps * sizeof(float));
	
	memset(h_times, 0, nreps *  sizeof(float));
	

	for(int k = 0; k < nreps; k++)
	{
		cudaEventRecord(start_event, 0);
		
				//we only transfer 1 byte.
				cudaMemcpy(d_data, h_data, sizeof(char), cudaMemcpyHostToDevice);
		
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&LoHTD, start_event, stop_event);
		
		h_times[k] = LoHTD;
	}
	
	LoHTD = getMedianTime(h_times, nreps);
	
	free(h_times);
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	
	return LoHTD;
}

/**
 * @brief Calculate PCIe DTH latency
 * @author Antonio Jose Lazaro Munoz
 * @date 17/02/2016
 * @details This function returns the latency of the PCIe for DTH memory transfers.
 * 
 * @param d_data Device data.
 * @param h_data Host data.
 * @param nreps iterations
 * @return DTH PCIe latency (ms).
 */
float getLoDTH(char *d_data, char *h_data, int nreps)
{

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	float LoDTH = 0.0;
	
	
	float *h_times = (float *)malloc(nreps * sizeof(float));
	
	memset(h_times, 0, nreps *  sizeof(float));
	
	for(int k = 0; k < nreps; k++)
	{
		cudaEventRecord(start_event, 0);
		
				//we only transfer 1 byte.
				cudaMemcpy(h_data, d_data, sizeof(char), cudaMemcpyDeviceToHost);
		
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&LoDTH, start_event, stop_event);
		
		h_times[k] = LoDTH;
	}
	
	LoDTH = getMedianTime(h_times, nreps);
	
	free(h_times);
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	
	return LoDTH;
}

/**
 * @brief DTH PCIe bandwidth (ms/byte).
 * @author Antonio Jose Lazaro Munoz
 * @date 17/02/2016
 * @details This function returns the bandwidth of the PCIe for DTH memory transfers.
 * 
 * @param d_data Device data.
 * @param h_data Host data.
 * @param LoDTH DTH PCIe latency.
 * @param nreps Iterations.
 * @return DTH PCIe bandwidth (ms/byte).
 */
float getGDTH(char *d_data, char *h_data, float LoDTH, int nreps)
{
	float time = 0;
	float GDTH = 0.0;
	float timeSumGDTH = 0.0;
	
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	int n = 0;
	int total_bytes = 0;

	//From 16 MB to 512 MB
	for(int size = 16; size <= 512; size=size*2)
	{
		
		cudaEventRecord(start_event, 0);
		
			cudaMemcpy(h_data, d_data, size * 1024 * 1024* sizeof(char), cudaMemcpyDeviceToHost);
	
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&time, start_event, stop_event);
		
		timeSumGDTH += time;
		total_bytes += size * 1024 * 1024;
		
		n++;
	}
	
	GDTH = (timeSumGDTH - n*LoDTH)/total_bytes;
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	
	return GDTH;
	
	
}

/**
 * @brief HTD PCIe bandwidth (ms/byte).
 * @author Antonio Jose Lazaro Munoz
 * @date 17/02/2016
 * @details This function returns the bandwidth of the PCIe for HTD memory transfers.
 * 
 * @param d_data Device data.
 * @param h_data Host data.
 * @param LoHTD HTD PCIe latency.
 * @param nreps Iterations.
 * @return HTD PCIe bandwidth (ms/byte).
 */
float getGHTD(char *d_data, char *h_data, float LoHTD, int nreps)
{
	float time = 0;
	float GHTD = 0.0;
	float timeSumGHTD = 0.0;
	
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	
	
	int n = 0;
	int total_bytes = 0;

	//From 16 MB to 512 MB
	for(int size = 16; size <= 512; size=size*2)
	{
		
		cudaEventRecord(start_event, 0);
		
			cudaMemcpy(d_data, h_data, size * 1024 * 1024* sizeof(char), cudaMemcpyHostToDevice);
	
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&time, start_event, stop_event);
		
		timeSumGHTD += time;
		total_bytes += size * 1024 * 1024;
		
		n++;
	}
	
	GHTD = (timeSumGHTD - n*LoHTD)/total_bytes;
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	
	
	return GHTD;
	
	
}

/**
 * @brief Overlap DTH PCIe bandwidth.
 * @author Antonio Jose Lazaro Munoz
 * @date 17/02/2016
 * @details This function returns the bandwidth of the PCIe for HTD memory transfers, when a DTH memory transfer
 * is concurrently executed.
 * 
 * @param d_data Device data.
 * @param h_data Host data.
 * @param LoDTH DTH PCIe latency.
 * @param nreps Iterations.
 * @return Overlap DTH PCIe bandwidth (byte/ms).
 */
float getOverlappedGDTH(char *d_data, char *h_data, float LoDTH, int nreps)
{
	float time = 0;
	float GDTH = 0.0;
	float timeSumGDTH = 0.0;
	
	cudaStream_t *stream_benchmark = (cudaStream_t *)malloc(2 * sizeof(cudaStream_t));
	
	for(int i = 0; i < 2; i++)
		cudaStreamCreate(&(stream_benchmark[i]));

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
		
	int n = 0;
	int total_bytes = 0;

	//From 16 MB to 512 MB
	for(int size = 16; size <= 512; size=size*2)
	{
		
		cudaEventRecord(start_event, stream_benchmark[0]);
		
		//DTH
		cudaMemcpyAsync(h_data, 
						d_data, 
						size * 1024 * 1024* sizeof(char), 
						cudaMemcpyDeviceToHost, stream_benchmark[0]);
	
		cudaEventRecord(stop_event, stream_benchmark[0]);
		
		
		//HTD
		cudaMemcpyAsync(d_data + (size * 1024 * 1024), 
						h_data + (size * 1024 * 1024), 
						size * 1024 * 1024* sizeof(char), 
						cudaMemcpyHostToDevice, stream_benchmark[1]);
		
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&time, start_event, stop_event);
		
		timeSumGDTH += time;
		total_bytes += size * 1024 * 1024;
		
		n++;
	}
	
	GDTH = (timeSumGDTH - n*LoDTH)/total_bytes;
	
	for(int i = 0; i < 2; i++)
		cudaStreamDestroy(stream_benchmark[i]);
	free(stream_benchmark);
	
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

	return GDTH;
	
	
}

/**
 * @brief Overlap HTD PCIe bandwidth.
 * @author Antonio Jose Lazaro Munoz
 * @date 17/02/2016
 * @details This function returns the bandwidth of the PCIe for DTH memory transfers, when a HTD memory transfer
 * is concurrently executed.
 * 
 * @param d_data Device data.
 * @param h_data Host data.
 * @param LoHTD HTD PCIe latency.
 * @param nreps Iterations.
 * @return Overlap HTD PCIe bandwidth (byte/ms).
 */
float getOverlappedGHTD(char *d_data, char *h_data, float LoHTD, int nreps)
{
	float time = 0;
	float GHTD = 0.0;
	float timeSumGHTD = 0.0;
	
	cudaStream_t *stream_benchmark = (cudaStream_t *)malloc(2 * sizeof(cudaStream_t));
	
	for(int i = 0; i < 2; i++)
		cudaStreamCreate(&(stream_benchmark[i]));
	
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	int n = 0;
	int total_bytes = 0;

	//From 16 MB to 512 MB
	for(int size = 16; size <= 512; size=size*2)
	{
		
		cudaEventRecord(start_event, stream_benchmark[0]);
		
		cudaMemcpyAsync(d_data, 
						h_data, 
						size * 1024 * 1024* sizeof(char), 
						cudaMemcpyHostToDevice, stream_benchmark[0]);
	
		cudaEventRecord(stop_event, stream_benchmark[0]);
		
		
		
		cudaMemcpyAsync(h_data + (size * 1024 * 1024), 
						d_data + (size * 1024 * 1024), 
						size * 1024 * 1024* sizeof(char), 
						cudaMemcpyDeviceToHost, stream_benchmark[1]);
		
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&time, start_event, stop_event);
		
		timeSumGHTD += time;
		total_bytes += size * 1024 * 1024;
		
		n++;
	}
	
	GHTD = (timeSumGHTD - n*LoHTD)/total_bytes;
	
	for(int i = 0; i < 2; i++)
		cudaStreamDestroy(stream_benchmark[i]);
	free(stream_benchmark);
	
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

	return GHTD;
	
	
}

/**
 * @brief PCIe microbenchmarking.
 * @author Antonio Jose Lazaro Munoz
 * @date 17/02/2016
 * @details This function calculates the values of PCIe latency and bandwidth 
 * for HTD and DTH memory transfers.
 * 
 * @param gpu GPU id.
 * @param LoHTD PCIe HTD latency (pointer).
 * @param LoDTH PCIe DTH latency (pointer).
 * @param GHTD HTD PCIe bandwidth (pointer).
 * @param overlappedGHTD Overlap HTD PCIe bandwidth (pointer).
 * @param GDTH DTH PCIe bandwidth (pointer).
 * @param overlappedGDTH Overlap DTH PCIe bandwidth (pointer).
 * @param nIter Iterations.
 */
void microbenchmarkingPCI(int gpu, float *LoHTD, float *LoDTH, float *GHTD, float *overlappedGHTD, 
					   float *GDTH, float *overlappedGDTH, int nIter)
{
	
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, gpu);
	cudaSetDevice(gpu);

	int tam = 1024*1024*1024;	// 1GB
	char *h_data_benchmark; cudaMallocHost((void**)&h_data_benchmark, tam * sizeof(char));
	memset(h_data_benchmark, 0, tam * sizeof(char));
		
	char *d_data_benchmark; cudaMalloc((void **)&d_data_benchmark, tam * sizeof(char));
	cudaMemset(d_data_benchmark, 0, tam * sizeof(char));
	
	*LoHTD = getLoHTD(d_data_benchmark, h_data_benchmark, nIter);
	*GHTD = getGHTD(d_data_benchmark, h_data_benchmark, *LoHTD, nIter);
		
	if(props.asyncEngineCount == 2)
		*overlappedGHTD = getOverlappedGHTD(d_data_benchmark, h_data_benchmark, *LoHTD, nIter);
			
	*LoDTH = getLoDTH(d_data_benchmark, h_data_benchmark, nIter);
	*GDTH = getGDTH(d_data_benchmark, h_data_benchmark, *LoDTH, nIter);
		
	if(props.asyncEngineCount == 2)
		*overlappedGDTH = getOverlappedGDTH(d_data_benchmark, h_data_benchmark, *LoDTH, nIter);
			
	 

	cudaFreeHost(h_data_benchmark);
	cudaFree(d_data_benchmark);
}