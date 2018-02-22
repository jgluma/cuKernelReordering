/**
 * @file microBenchmarking-Transfers.cu
 * @details This file describes the functions belonging to MicroBenchmarkingTransfers class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 11/11/2017
 */
#include <microBenchmarking-Transfers.h>

MicroBenchmarkingTransfers::MicroBenchmarkingTransfers(int gpuid, int min_Bytes, int max_Bytes)
{
	  gpuId           = gpuid;
	  LoHTD           = 0.0;
    LoDTH           = 0.0;
    GHTD            = 0.0;
    overlappedGHTD  = 0.0;
    GDTH            = 0.0;
    overlappedGDTH  = 0.0;
    nIter		        = 15;
    min_size_Bytes  = min_Bytes * 1024 * 1024;
  	max_size_Bytes  = max_Bytes * 1024 * 1024;
  	increment_bytes = 2 * 1024 * 1024;

    //Cuda setting enviroment
	int num_devices=0;
	cudaGetDeviceCount(&num_devices);
	if(0==num_devices){
		cout << "your system does not have a CUDA capable device" << endl;
		exit(EXIT_FAILURE);
	}
	
	if(num_devices <= gpuId)
	{
		cout << "GPU id is wrong" << endl;
		exit(EXIT_FAILURE);
	}


	cudaGetDeviceProperties(&device_properties, gpuId);
	if( (1 == device_properties.major) && (device_properties.minor < 1))
		cout << device_properties.name << " does not have compute capability 1.1 or later\n" << endl;

	gpuName = device_properties.name;
	
	cudaSetDevice(gpuId);

	//We creating the CUDA streams
	stream_benchmark = new cudaStream_t[2];

	for(int i = 0; i < 2; i++)
		cudaStreamCreate(&(stream_benchmark[i]));
}

void MicroBenchmarkingTransfers::computeLoHTD(void)
{
#ifdef STANDARD

	//Host vector.
     char * h_data = (char *)malloc(sizeof(char));

#endif

#ifdef PINNED 
     //Host vector
     char *h_data; cudaMallocHost((void **)&h_data, sizeof(char));
#endif

     //Device Vector.
     char *d_data; cudaMalloc((void **)&d_data, sizeof(char));
     vector<float>time_loHTD_rep;

     //Cuda events
     cudaEvent_t start_event, stop_event;

     //Creating CUDA events
     cudaEventCreate(&start_event);
	 cudaEventCreate(&stop_event);

	 //Init host data
     h_data[0] = (char)1;

  	 for(int iter = 0; iter < nIter; iter++)
     {
     	cudaEventRecord(start_event, 0);
		
				cudaMemcpyAsync(d_data, h_data, sizeof(char), cudaMemcpyHostToDevice, stream_benchmark[0]);
		
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		float elapsed_time;
		cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
		time_loHTD_rep.push_back(elapsed_time);


     }

     //Get median time
      LoHTD = getMedian(time_loHTD_rep); //CPU timer includes enqueue, submit and execution time

#ifdef STANDARD
      free(h_data);
#endif

#ifdef PINNED 
      cudaFreeHost(h_data);
#endif

      //Free device memory
      cudaFree(d_data);

      //Destroying CUDA events
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);

}

void MicroBenchmarkingTransfers::computeLoDTH(void)
{
#ifdef STANDARD

	//Host vector.
     char * h_data = (char *)malloc(sizeof(char));

#endif

#ifdef PINNED 
     //Host vector
     char *h_data; cudaMallocHost((void **)&h_data, sizeof(char));
#endif

     //Device Vector.
     char *d_data; cudaMalloc((void **)&d_data, sizeof(char));
     vector<float>time_loDTH_rep;

     //Cuda events
     cudaEvent_t start_event, stop_event;

     //Creating CUDA events
     cudaEventCreate(&start_event);
	 cudaEventCreate(&stop_event);

	 //Init host data
     h_data[0] = (char)1;
     //Init device data
     cudaMemcpyAsync(d_data, h_data, sizeof(char), cudaMemcpyHostToDevice, stream_benchmark[0]);


  	 for(int iter = 0; iter < nIter; iter++)
     {
     	cudaEventRecord(start_event, 0);
		
				cudaMemcpyAsync(h_data, d_data, sizeof(char), cudaMemcpyDeviceToHost, stream_benchmark[0]);
		
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		float elapsed_time;
		cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
		time_loDTH_rep.push_back(elapsed_time);


     }

     //Get median time
      LoDTH = getMedian(time_loDTH_rep); //CPU timer includes enqueue, submit and execution time

#ifdef STANDARD
      free(h_data);
#endif

#ifdef PINNED 
      cudaFreeHost(h_data);
#endif

      //Free device memory
      cudaFree(d_data);

      //Destroying CUDA events
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);

}

float MicroBenchmarkingTransfers::getMedian(vector<float> v)
{
  vector<float>v_c;
  float median;
  v_c = v;


  sort(v_c.begin(), v_c.end());

  if(v_c.size()%2)
    median = (v_c.at(v_c.size()/2) + v_c.at(v_c.size()/2 - 1))/2;
  else
    median = v_c.at((int)v_c.size()/2);

  return median;
}

void MicroBenchmarkingTransfers::computeGHTD(void)
{
	
  float timeSumGHTD = 0.0;
  
  //Cuda events
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  
  
  int n = 0;
  int total_bytes = 0;

  for(int size = min_size_Bytes; size <= max_size_Bytes; size=size*2)
  {
#ifdef STANDARD
      //Host vector.
      char * h_data = (char *)malloc(size * sizeof(char));
#endif

#ifdef  PINNED
      //Host vector
      char *h_data; cudaMallocHost((void **)&h_data, size * sizeof(char));
#endif
      //Device vector
      char *d_data; cudaMalloc((void **)&d_data, size * sizeof(char));

      //Init host data
      for(int i = 0; i < size; i++)
        h_data[i] = (char)i;

      vector<float> v_time_iter;

      for(int iter = 0; iter < nIter; iter++)
      {
        cudaEventRecord(start_event, 0);
    
          cudaMemcpy(d_data, h_data, size * sizeof(char), cudaMemcpyHostToDevice);
    
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        v_time_iter.push_back(elapsed_time);


      }

      float time_htd = getMedian(v_time_iter);
      v_time_htd.push_back(time_htd);

      timeSumGHTD += time_htd;
      total_bytes += size;
      
      n++;

#ifdef STANDARD
      free(h_data);
#endif

#ifdef PINNED
      cudaFreeHost(h_data);
#endif

      cudaFree(d_data);

    }

    GHTD = (timeSumGHTD - n*LoHTD)/total_bytes;

    //Destroying CUDA events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

}

void MicroBenchmarkingTransfers::computeGDTH(void)
{
  
 float timeSumGDTH = 0.0;
  
  //Cuda events
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  
  
  int n = 0;
  int total_bytes = 0;

  for(int size = min_size_Bytes; size <= max_size_Bytes; size=size*2)
  {
#ifdef STANDARD
      //Host vector.
      char * h_data = (char *)malloc(size * sizeof(char));
#endif

#ifdef  PINNED
      //Host vector
      char *h_data; cudaMallocHost((void **)&h_data, size * sizeof(char));
#endif
      //Device vector
      char *d_data; cudaMalloc((void **)&d_data, size * sizeof(char));

      //Init host data
      for(int i = 0; i < size; i++)
        h_data[i] = (char)i;

      //Init device data
      cudaMemcpy(d_data, h_data, size * sizeof(char), cudaMemcpyHostToDevice);

      vector<float> v_time_iter;

      for(int iter = 0; iter < nIter; iter++)
      {
        cudaEventRecord(start_event, 0);
    
          cudaMemcpy(h_data, d_data, size * sizeof(char), cudaMemcpyDeviceToHost);
    
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        v_time_iter.push_back(elapsed_time);


      }

      float time_dth = getMedian(v_time_iter);
      v_time_dth.push_back(time_dth);

      timeSumGDTH += time_dth;
      total_bytes += size;
      
      n++;

#ifdef STANDARD
      free(h_data);
#endif

#ifdef PINNED
      cudaFreeHost(h_data);
#endif

      cudaFree(d_data);

    }

    GDTH = (timeSumGDTH - n*LoDTH)/total_bytes;

    //Destroying CUDA events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

}

void MicroBenchmarkingTransfers::computeOverlappedG(void)
{
  
 
  float timeSumGHTD = 0.0;
  float timeSumGDTH = 0.0;
  
    
  int n = 0;
  int total_bytes = 0;

  //Cuda events
  cudaEvent_t start_event_htd, stop_event_htd;
  cudaEvent_t start_event_dth, stop_event_dth;

  //Creating CUDA events
  cudaEventCreate(&start_event_htd);
  cudaEventCreate(&stop_event_htd);
  cudaEventCreate(&start_event_dth);
  cudaEventCreate(&stop_event_dth);

  for(int size = min_size_Bytes; size <= max_size_Bytes; size=size*2)
  {

#ifdef STANDARD
  //Host vectors
    char * h_data_htd = (char *)malloc(size * sizeof(char));
    char * h_data_dth = (char *)malloc(size * sizeof(char));
#endif

#ifdef PINNED
    char * h_data_htd; cudaMallocHost((void **)&h_data_htd, size * sizeof(char));
    char * h_data_dth; cudaMallocHost((void **)&h_data_dth, size * sizeof(char));
#endif

    char *d_data_htd; cudaMalloc((void **)&d_data_htd, size * sizeof(char));
    char *d_data_dth; cudaMalloc((void **)&d_data_dth, size * sizeof(char));

    //Init host vector
    for(int i = 0; i < size; i++)
    {
      h_data_htd[i] = (char)i;
      h_data_dth[i] = (char)i;
    }

    vector<float> v_time_iter_htd;
    vector<float> v_time_iter_dth;
    //Init device vector
    cudaMemcpy(d_data_dth, h_data_dth, size * sizeof(char), cudaMemcpyHostToDevice);

    for(int iter = 0; iter < nIter; iter++)
    {
      //HTD
      cudaEventRecord(start_event_htd, stream_benchmark[0]);
    
      cudaMemcpyAsync(d_data_htd, 
                    h_data_htd, 
                    size * sizeof(char), 
                    cudaMemcpyHostToDevice, stream_benchmark[0]);
  
      cudaEventRecord(stop_event_htd, stream_benchmark[0]);


      //DTH
      cudaEventRecord(start_event_dth, stream_benchmark[1]);

      cudaMemcpyAsync(h_data_dth, 
                    d_data_dth, 
                    size * sizeof(char), 
                    cudaMemcpyDeviceToHost, stream_benchmark[1]);

      cudaEventRecord(stop_event_dth, stream_benchmark[1]);


      cudaDeviceSynchronize();
      float elapsed_time_htd;
      float elapsed_time_dth;
      cudaEventElapsedTime(&elapsed_time_htd, start_event_htd, stop_event_htd);
      v_time_iter_htd.push_back(elapsed_time_htd);
      cudaEventElapsedTime(&elapsed_time_dth, start_event_dth, stop_event_dth);
      v_time_iter_dth.push_back(elapsed_time_dth);

    }

    float time_ohtd = getMedian(v_time_iter_htd);
    float time_odth = getMedian(v_time_iter_dth);
    v_time_ohtd.push_back(time_ohtd);
    v_time_odth.push_back(time_odth);

    timeSumGHTD += time_ohtd;
    timeSumGDTH += time_odth;
    total_bytes += size;
      
    n++;



#ifdef  STANDARD
    free(h_data_htd);
    free(h_data_dth);
#endif

#ifdef PINNED
    cudaFreeHost(h_data_htd);
    cudaFreeHost(h_data_dth);
#endif

    cudaFree(d_data_htd);
    cudaFree(d_data_dth);

  }

  overlappedGHTD = (timeSumGHTD - n*LoHTD)/total_bytes;
  overlappedGDTH = (timeSumGDTH - n*LoDTH)/total_bytes;

  //Destroying CUDA events
  cudaEventDestroy(start_event_htd);
  cudaEventDestroy(stop_event_htd);
  cudaEventDestroy(start_event_dth);
  cudaEventDestroy(stop_event_dth);

}

void MicroBenchmarkingTransfers::execute(void)
{
	//LoHTD
	computeLoHTD();
	//LoDTH
	computeLoDTH();
  //GHTD
  computeGHTD();
  //GDTH
  computeGDTH();
  //Compute the overlapped bandwidth for the HTD and DTH transfers.
  computeOverlappedG();
  
}

MicroBenchmarkingTransfers::~MicroBenchmarkingTransfers()
{
	//Detroying CUDA streams
	for(int i = 0; i < 2; i++)
		cudaStreamDestroy(stream_benchmark[i]);
	delete [] stream_benchmark;
}