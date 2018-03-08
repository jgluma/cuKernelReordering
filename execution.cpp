#define N_TASKS 4
#include <iostream>
#include <thread>
#include "buffer.h"
#include "batch.h"
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <fstream>
#include <algorithm>
#include <vector>
#include "microbenchmarking_transfers.h"
#include "TaskTemporizer.h"
#include <sys/time.h>
#include <unistd.h> 
#include <chrono>

#define MM   0
#define HST  1
#define BS   2
#define GS   3
#define CONV 4
#define TM   5
#define FWT  6
#define ND   7
#define PAF  8
#define PF   9
#define VA   10
#define MYFFT  11
#define ROTM 12
#define SB   13

#include "vectorAdd.h"
#include "matrixMult.h"
#include "microbenchmarking_transfers.h"
#include "histogram.h"
#include "transpose.h"
#include "BlackScholes.h"
#include "FastWalshTransform.h"
#include "ConvolutionSeparable.h"
#include "Needle.h"
#include "Gaussian.h"
#include "ParticleFilter.h"
#include "PathFinder.h"
#include "SobolQRNG.h"

using namespace std;

#define LAUNCH_LAST_HTD	1
#define LAUNCH_BATCH	0

#define DK	0
#define DT	1

/**
 * @brief      Tokenizer
 * @details    This function tokenizes a string according to a delimiter.
 * @author     Antonio Jose Lazaro Munoz.
 * @date       15/05/2017
 *
 * @param[in]  str        The string.
 * @param[in]  delimiter  The delimiter.
 *
 * @return     strings array with the tokens
 */
vector<string> tokenize(const string& str,const string& delimiters)
{
	vector<string> tokens;
    	
	// skip delimiters at beginning.
    	string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    	
	// find first "non-delimiter".
    	string::size_type pos = str.find_first_of(delimiters, lastPos);

    	while (string::npos != pos || string::npos != lastPos)
    	{
        	// found a token, add it to the vector.
        	tokens.push_back(str.substr(lastPos, pos - lastPos));
		
        	// skip delimiters.  Note the "not_of"
        	lastPos = str.find_first_not_of(delimiters, pos);
		
        	// find next "non-delimiter"
        	pos = str.find_first_of(delimiters, lastPos);
    	}

	return tokens;
};

void swapG(float & v1, float & v2){
    float tmp = v1;
    v1 = v2;
    v2 = tmp;
}

int partitionG(float *array, int left, int right){
    int part = right;
    swapG(array[part],array[(right+left) / 2]);
    
    --right;
 
    while(true){
        while(array[left] < array[part]){
            ++left;
        }
        while(right >= left && array[part] <= array[right]){
            --right;
        }
        if(right < left) break;
 
        swapG(array[left],array[right]);
        ++left;
        --right;
    }
 
    swap(array[part],array[left]);
 
    return left;
}

void qsG(float * array, const int left, const int right){
    if(left < right){
        const int part = partitionG(array, left, right);
        qsG(array, part + 1,right);
        qsG(array, left,part - 1);
    }
}

void serialQuickSortG(float *array, const int size){
    qsG(array, 0,size-1);
}

float getMedianTimeG(float *h_times, int N)
{
	float median = 0;

	float * h_sorted_times = (float *)malloc(N * sizeof(float));
	
	for(int n = 0; n < N; n++)
		h_sorted_times[n] = h_times[n];
		
	//Sort execution times
	serialQuickSortG(h_sorted_times, N);
	
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

void process(int* P, int N, int* total, int *permutaciones, int per) { 
	//cout << "\tEntra" << endl;
	//printf("Permutacion: %d\n", per);   
	int i;
 
	for (i=N; i>0; i--) {
		//printf("%d\n", P[i]-1);
		permutaciones[per*N + N-i] = P[i]-1;
	}

	// printf("\n");
	(*total) ++;
}


/* esto seguro que sabes lo que hace ... */
void swap(int *x, int *y) { 
	int temp = *x;
	*x = *y;
	*y = temp;
}

/* ==== Comienzo de la magia ==== 
 * Extraido de R. Sedgewick, "Permutations Generation Methods"
 * ACM Computing Surveys, Vol. 9, No. 2, p. 154, Junio 1977
 */

/* invierte el orden del array de enteros P */
void reverse (int *P, int N) {
	int i = 1;  
	while ( i < (N+1-i) ) {
		swap(&P[i], &P[N+1-i]);
		i ++;
	}
}

int B(int N, int c) {  
	return ( (N % 2) != 0 ? 1 : c );
}

void lexperms (int *P, int N, int *total, int *permutaciones) {
	int i;
	int per = 0;
	int c[N];
	
	for (i = N; i > 1; i --) {
		c[i] = 1;
	}
	
	i = 2;
  
	// cout << "process:" << endl;
	process(P,N,total, permutaciones, per);
	// cout << "end process" << endl;
	per++;
   
	do {
		if (c[i] < i) {
			swap(&P[i],&P[c[i]]);
			reverse(P,i-1); /* inversion parcial! */
			process(P,N,total, permutaciones, per);
			per++;

			c[i] ++;
			i = 2;
		} else {
			c[i] = 1;
			i ++;
		}
	} while (i <= N);
}

long factorial(int n) {
   if(n < 0) return 0;
   else if(n > 1) return n*factorial(n-1); 
   return 1; 
}

/**
 * @brief      Producer thread function
 * @details    This function implements a producer thread. A producer thread inserts a number of task equal to nepoch. 
 * These tasks are pushed into a shared buffer among the producers and proxy thread. Each task is inserted according to 
 * a waiting time.
 * @author     Antonio Jose Lazaro Munoz.
 * @date       15/05/2017
 *  
 * @param[in]  gpu                		The gpu id
 * @param[in]  tid                		The thread id
 * @param[in]  producer_buffers   		The shared producer buffers
 * @param[in]  nepoch             	  	The number of epochs
 * @param[in]  id_launched_tasks  		Ids array of the tasks to insert.
 * @param[in]  waiting_times      		Waiting time array.
 */
void producerThread(int gpu, int tid, vector<BufferTasks> &producer_buffers, int nepoch, int *id_launched_tasks,
	float *waiting_times, atomic<int> &sync_producers, int nproducer)
{
	infoTask task;

	srand(tid);

	//Inserting tasks
	for(int iter = 0; iter < nepoch; iter++)
	{	
		float pause_time = waiting_times[tid*nepoch + iter];

		usleep((int)(pause_time*1000));

		task.id_thread = tid;
		task.id_task = id_launched_tasks[tid*nepoch + iter];

		producer_buffers[tid].pushBack(task);		
	}

	//Inserting STOP task
	task.id_thread = tid;
	task.id_task = STOP;
			
	producer_buffers[tid].pushBack(task);
}

bool proxy_shutdown(vector <bool> v, int n)
{
	bool exit = true;
	
	for(int p = 0; p < n; p++)
	{
		if(v[p] == false)
			exit = false;
	}
	
	return exit;
}

void allocHostMemory(int n_app, ifstream &fb, 
					vector <Task *>&tasks_v, int gpu)
{
	

	for(int app = 0; app < n_app; app++)
	{	
		string line;
		
		getline(fb, line);
		
		vector<string> v(tokenize(line, "\t"));

		int type_task = atoi(v[0].c_str());
		int num_param = atoi(v[1].c_str());
		
		switch(type_task)
		{
			case MM:
			{
				int *sizes = new int[num_param];

				for(int i = 0; i < num_param; i++)
					sizes[i] = atoi(v[i+2].c_str());
				
				//Creamos la tarea
				tasks_v.push_back(new MatrixMult(sizes));

				delete [] sizes;

				break;
			}

			case HST:
			{
				int nframes = atoi(v[2].c_str());

				tasks_v.push_back(new Histogram(nframes));

				break;
			}

			case CONV:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new ConvolutionSeparable(params[0], params[1], params[2]));

				delete [] params;

				break;
			}

			case TM:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new Transpose(params[0], params[1]));

				delete [] params;
				break;
			}

			case FWT:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new FastWalshTransform(params[0], params[1]));

				delete [] params;
				break;
			}

			case BS:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new BlackScholes(params[0], params[1]));

				delete [] params;

				break;
			}

			case GS:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new Gaussian(params[0]));

				delete [] params;

				break;
			}

			case ND:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new Needle(params[0], params[1], params[2]));

				delete [] params;

				break;
			}

			case PAF:	//ParticleFilter
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new ParticleFilter(params[0], params[1], params[2], params[3]));

				delete [] params;

				break;
			}

			case PF: //PathFinder
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new PathFinder(params[0], params[1], params[2]));

				delete [] params;

				break;
			}

			case VA:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new VectorADD(params[0]));

				delete [] params;

				break;
			}

			case SB:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new SobolQRNG(params[0], params[1], gpu));

				delete [] params;

				break;
			}
		}

		//Alloc host memory
		tasks_v.at(app)->allocHostMemory();
	}
}

void allocDeviceMemory(int n_app, vector<Task *> &tasks_v)
{
	for(int app = 0; app < n_app; app++)
	{
		//Alloc device memory
		tasks_v.at(app)->allocDeviceMemory();
	}
}

void generatingData(int n_app, vector<Task *> &tasks_v)
{
	int idx_vector_SK = 0;
	
	for(int app = 0; app < n_app; app++)
	{
		//Generating input data
		tasks_v.at(app)->generatingData();
	}
}

void freeHostMemory(int n_app, vector <Task *> &tasks_v)
{
	for(int app = 0; app < n_app; app++)
	{
		tasks_v.at(app)->freeHostMemory();
	}
}

void freeDeviceMemory(int n_app, vector<Task *> &tasks_v)
{
	for(int app = 0; app < n_app; app++)
	{
		
		tasks_v.at(app)->freeDeviceMemory();
	}
}

void launching_applications_2CopyEngines_SIN_HYPERQ(cudaStream_t *streams, int *idx_processes,
										int nstreams, int n_app, Batch execution_batch,
										cudaEvent_t *evt_finish, cudaEvent_t *evt_finish_launch, 
										vector <Task *> &tasks_v, int nepoch)
{
	int t = 0;	
	for(int i = 0; i < n_app; i++)
	{
		int tid = idx_processes[i];
		
		int id_task = execution_batch.getProcessTaskBatch(tid);
		
		//if(launching_per_thread[tid] == 0)
			//task_timer[tid].setTaskLaunchTime();

		//launching_per_thread[tid]++;

#if PRINT_GPU_TRACE					
			cout << "PROCESO: " << tid << endl;
#endif

			//cout << "Elapsed Time: " << elapsed_time << endl;
		
#if PRINT_GPU_TRACE						
						cout << "\t\t\tLanzando SK - Stream: " << tid << " - HTD percent: " << HTD_percentage[id_task] << " - Computation: " << computation_percentage[id_task] << " - DTH percent: " << DTH_percentage[id_task] << endl;
#endif
						
							//memHostToDevice_SK(streams[idx_process], task_offset, HTD_percentage[id_task]);
							tasks_v.at(id_task)->memHostToDeviceAsync(streams[tid]);

#if LAUNCH_LAST_HTD
						if(t == n_app - 1)
							cudaEventRecord(*evt_finish, streams[tid]);
#endif							
						
							tasks_v.at(id_task)->launch_kernel_Async(streams[tid]);

							//launch_SK(streams[idx_process], task_offset, computation_percentage[id_task]);
						
							tasks_v.at(id_task)->memDeviceToHostAsync(streams[tid]);

							//memDeviceToHost_SK(streams[idx_process], task_offset, DTH_percentage[id_task]);	
				
						if(t == n_app - 1)
							cudaEventRecord(*evt_finish_launch, streams[tid]);
									
#if LAUNCH_BATCH
						if(t == n_app - 1)
						{
							cudaEventRecord(*evt_finish, streams[tid]);
						}
#endif

		t++;			
	}
}

void launching_applications_2CopyEngines(cudaStream_t *streams, int *idx_processes,
										int nstreams, int n_app, Batch execution_batch,
										float *HTD_percentage, float *computation_percentage, float *DTH_percentage, 
										cudaEvent_t *evt_finish, cudaEvent_t *evt_finish_launch, 
										vector <Task *> &tasks_v,
										cudaEvent_t *htd_end, cudaEvent_t *dth_end,
										int nepoch)
{
	int t = 0;	
	for(int i = 0; i < n_app; i++)
	{
		int tid = idx_processes[i];
		
		int id_task = execution_batch.getProcessTaskBatch(tid);

		//Si no es la primera HTD, entonces se espera a que la HTD anterior termine.
		if(i != 0)
			cudaStreamWaitEvent(streams[tid], htd_end[idx_processes[i-1]], 0);

		tasks_v.at(id_task)->memHostToDeviceAsync(streams[tid]);

		cudaEventRecord(htd_end[tid], streams[tid]);

#if LAUNCH_LAST_HTD
		if(t == n_app - 1)
			cudaEventRecord(*evt_finish, streams[tid]);
#endif							
									
		tasks_v.at(id_task)->launch_kernel_Async(streams[tid]);
				
		//Si es la primera DTH se espera a que la ultima DTH del ultimo
		//Lanzamiento se haya realizado
		if(i == 0)
			cudaStreamWaitEvent(streams[tid], *evt_finish_launch, 0);
		
		//Si no es la primera DTH se espera que la DTH anterior termine.
		if(i != 0)
			cudaStreamWaitEvent(streams[tid], dth_end[idx_processes[i-1]], 0);
						
		tasks_v.at(id_task)->memDeviceToHostAsync(streams[tid]);
			
		cudaEventRecord(dth_end[tid], streams[tid]);
				
		if(i == n_app - 1)
			cudaEventRecord(*evt_finish_launch, streams[tid]);
									
#if LAUNCH_BATCH
		if(t == n_app - 1)
		{
			cudaEventRecord(*evt_finish, streams[tid]);
		}
#endif
		t++;			
	}
}

void handler_gpu_func(int gpu, atomic<int> &stop_handler_gpu, BufferTasks &pending_tasks_buffer, int max_tam_batch, ifstream &fb,
	int nstreams, int nepoch, atomic<int> &init_proxy, int iter, int nIter, float *elapsed_times, 
	int *n_launching_tasks,
	vector<float>&scheduling_times, int *selected_order)
{
	//GPU select
	cudaSetDevice(gpu);
	//Scheduling batch.
	Batch scheduling_batch(max_tam_batch);
	//Execution batch.
	Batch execution_batch(max_tam_batch);

	int scheduled_tasks = 0;

	//CUDA streams
	cudaStream_t *streams = new cudaStream_t[nstreams];	//nstreams = nproducers
	for(int i = 0; i < nstreams; i++)
	  cudaStreamCreate(&(streams[i]));

	//Launching order vector.
  	int *h_order_processes = new int [nstreams];
  	//Processes order vector in the execute batch.
  	int *execute_batch = new int [nstreams];

  	//CUDA event default stream
  	cudaEvent_t start_event, stop_event;
  
  	cudaEventCreate(&start_event);
  	cudaEventCreate(&stop_event);
  
  	cudaEvent_t evt_finish;				//CUDA event for the last HTD belonging to the current epoch.
  	cudaEvent_t evt_finish_launch;		//CUDA event for the last DTH belonging to the previous epoch.
  
  	cudaEventCreate(&evt_finish);
  	cudaEventCreate(&evt_finish_launch);
  
  	//CUDA event for the HTD commands.
  	cudaEvent_t *htd_end =  (cudaEvent_t *)malloc(nstreams*sizeof(cudaEvent_t));
  	//CUDA event for the DTH commands.
  	cudaEvent_t *dth_end =  (cudaEvent_t *)malloc(nstreams*sizeof(cudaEvent_t));
  	//CUDA event for the Kernel commands.
  	cudaEvent_t *kernel_end =  (cudaEvent_t *)malloc(nstreams*sizeof(cudaEvent_t));

  	//Creamos los eventos para sincronizar las HTD
  	for(int i = 0; i < nstreams; i++)
  	{
		cudaEventCreate(&htd_end[i]);
		cudaEventCreate(&dth_end[i]);
		cudaEventCreate(&kernel_end[i]);
  	}
	
	fb.seekg (0, fb.beg);

  	//Vector synthetic tasks
	vector <Task *> tasks_v;

	//Alloc Host memory
	
  	allocHostMemory(N_TASKS, fb, tasks_v, gpu);

  	//Alloc Device memory
  	allocDeviceMemory(N_TASKS, tasks_v);

  	//Generating data
  	generatingData(N_TASKS, tasks_v);
  	fb.seekg (0, fb.beg);  	

	//CPU timers
	struct timeval t1, t2; //, t1_perm, t2_perm, t1_creation_perm, t2_creation_perm, t1_total_sim_perm, t2_total_sim_perm;
  	struct timeval t1_scheduling, t2_scheduling;
  	float elapsed_time_perm;
  	float elapsed_creation_perm;
  	float elapsed_total_sim_perm;

  	init_proxy.store(1);

  	gettimeofday(&t1, NULL);
  	bool start_work = false;

  	cudaProfilerStart();

  	//Esperamos a que todas las tareas esten disponibles en el buffer
  	while(pending_tasks_buffer.getProducedElements() != N_TASKS*nepoch);

	for(int epoch = 0; epoch < nepoch; epoch++)
	{
		int n_pending_tasks = pending_tasks_buffer.getProducedElements();
		//Si tenemos tareas pendientes por consumir
		if(n_pending_tasks > 0)
		{
			vector <infoTask> inserted_tasks;

			//Looping the pending tasks buffer.			
			for(int i = 0; i < pending_tasks_buffer.getProducedElements(); i++)
			{
				infoTask task;

				pending_tasks_buffer.getTask(task, i);

				//We try to insert the tasks into the scheduling batch.
				if(scheduling_batch.insertTask(task.id_thread, task.id_task))
				{
					inserted_tasks.push_back(task);
				}
			}
			
			//We delete the pending tasks which have been inserted into the scheduling batch.
			pending_tasks_buffer.deleteSetTasks(inserted_tasks);
			
			n_launching_tasks[scheduling_batch.getTamBatch()-1]++;
											
			execution_batch.cleanBatch();
			execution_batch.copyBatch(scheduling_batch);

			for(int app = 0; app < N_TASKS; app++)
				h_order_processes[app] = selected_order[epoch * N_TASKS + app];
			
			//Lanzamos la epoca actual con su orden 
			launching_applications_2CopyEngines_SIN_HYPERQ(streams, h_order_processes,
														nstreams, execution_batch.getTamBatch(), execution_batch,
														&evt_finish, &evt_finish_launch, tasks_v, 1);
				
			//Limpiando tareas del batch
			scheduled_tasks+=scheduling_batch.getTamBatch();
			scheduling_batch.cleanBatch();

			//Esperamos a que termine el ultimo HTD para volver a planificar
			while(cudaEventQuery(evt_finish) != cudaSuccess);
		}
	}

	cudaDeviceSynchronize();

	cudaProfilerStop();

	gettimeofday(&t2, NULL);
	double timer = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);
	float elapsed_time = timer/1000.0;

	elapsed_times[iter] = elapsed_time;	

	for(int t = 0; t < N_TASKS; t++)
  		tasks_v.at(t)->checkResults();

	//Free host memory
  	freeHostMemory(N_TASKS, tasks_v);

  	//Free device memory
  	freeDeviceMemory(N_TASKS, tasks_v);

	for(int i = 0; i < nstreams; i++)
		cudaStreamDestroy(streams[i]);
	delete [] streams;

	delete [] h_order_processes;
  	delete [] execute_batch;
  	
  	//Destroying tasks
  	for(int t = 0; t < N_TASKS; t++)
  		tasks_v.pop_back();

  	cudaEventDestroy(evt_finish);
  	cudaEventDestroy(evt_finish_launch);
  	cudaEventDestroy(start_event);
  	cudaEventDestroy(stop_event);
  
  	for(int i = 0; i < nstreams; i++)
  	{
		cudaEventDestroy(htd_end[i]);
		cudaEventDestroy(dth_end[i]);
		cudaEventDestroy(kernel_end[i]);
	}
 	
	free(htd_end);
	free(dth_end);
	free(kernel_end);
}

/**
 * @brief      Sets the File name 4 producer.
 * @details    This function sets the file name of the task id when the number of the producer threads are 4. 
 * The name of the file is created
 * according to the benchmar id
 * @author     Antonio Jose Lazaro Munoz.
 * @date       15/05/2017
 *
 * @param[in]      name       String reference to the file name.
 * @param[in]      benchmark  Benchmark id.
 */
void setFileName_4producer(string &name, int benchmark)
{
	switch(benchmark)
	{
		case 1:
		{
			name = "ID-Tasks-1DK-3DT.txt";
			break;
		}

		case 2:
		{
			name = "ID-Tasks-3DK-1DT.txt";
			break;
		}

		case 3:
		{
			name = "ID-Tasks-2DK-2DT.txt";
			break;
		}

		case 4:
		{
			name = "ID-Tasks-4DK-0DT.txt";
			break;
		}

		case 5:
		{
			name = "ID-Tasks-0DK-4DT.txt";
			break;
		}
	}
}

/**
 * @brief      Sets the File name 8 producer.
 * @details    This function sets the file name of the task id when the number of the producer threads are 8. 
 * The name of the file is created
 * according to the benchmar id
 * @author     Antonio Jose Lazaro Munoz.
 * @date       15/05/2017
 *
 * @param[in]      name       String reference to the file name.
 * @param[in]      benchmark  Benchmark id.
 */
void setFileName_8producer(string &name, int benchmark)
{
	switch(benchmark)
	{
		case 1:
		{
			name = "ID-Tasks-3DK-5DT.txt";
			break;
		}

		case 2:
		{
			name = "ID-Tasks-5DK-3DT.txt";
			break;
		}

		case 3:
		{
			name = "ID-Tasks-4DK-4DT.txt";
			break;
		}

		case 4:
		{
			name = "ID-Tasks-8DK-0DT.txt";
			break;
		}

		case 5:
		{
			name = "ID-Tasks-0DK-8DT.txt";
			break;
		}
	}
}

/**
 * @brief      Sets the File name 6 producer.
 * @details    This function sets the file name of the task id when the number of the producer threads are 6. 
 * The name of the file is created
 * according to the benchmar id
 * @author     Antonio Jose Lazaro Munoz.
 * @date       15/05/2017
 *
 * @param[in]      name       String reference to the file name.
 * @param[in]      benchmark  Benchmark id.
 */
void setFileName_6producer(string &name, int benchmark)
{
	switch(benchmark)
	{
		case 1:
		{
			name = "ID-Tasks-2DK-4DT.txt";
			break;
		}

		case 2:
		{
			name = "ID-Tasks-4DK-2DT.txt";
			break;
		}

		case 3:
		{
			name = "ID-Tasks-3DK-3DT.txt";
			break;
		}

		case 4:
		{
			name = "ID-Tasks-6DK-0DT.txt";
			break;
		}

		case 5:
		{
			name = "ID-Tasks-0DK-6DT.txt";
			break;
		}
	}
}

/**
 * @brief      Sets the File name 16 producer.
 * @details    This function sets the file name of the task id when the number of the producer threads are 16. 
 * The name of the file is created
 * according to the benchmar id
 * @author     Antonio Jose Lazaro Munoz.
 * @date       15/05/2017
 *
 * @param[in]      name       String reference to the file name.
 * @param[in]      benchmark  Benchmark id.
 */
void setFileName_16producer(string &name, int benchmark)
{
	switch(benchmark)
	{
		case 1:
		{
			name = "ID-Tasks-4DK-12DT.txt";
			break;
		}

		case 2:
		{
			name = "ID-Tasks-6DK-10DT.txt";
			break;
		}

		case 3:
		{
			name = "ID-Tasks-8DK-8DT.txt";
			break;
		}

		case 4:
		{
			name = "ID-Tasks-16DK-0DT.txt";
			break;
		}

		case 5:
		{
			name = "ID-Tasks-16DK-0DT.txt";
			break;
		}
	}
}

/**
 * @brief      Sets the File name 32 producer.
 * @details    This function sets the file name of the task id when the number of the producer threads are 32. 
 * The name of the file is created
 * according to the benchmar id
 * @author     Antonio Jose Lazaro Munoz.
 * @date       15/05/2017
 *
 * @param[in]      name       String reference to the file name.
 * @param[in]      benchmark  Benchmark id.
 */
void setFileName_32producer(string &name, int benchmark)
{
	switch(benchmark)
	{
		case 1:
		{
			name = "ID-Tasks-4DK-28DT.txt";
			break;
		}

		case 2:
		{
			name = "ID-Tasks-6DK-26DT.txt";
			break;
		}

		case 3:
		{
			name = "ID-Tasks-8DK-24DT.txt";
			break;
		}

		case 4:
		{
			name = "ID-Tasks-16DK-16DT.txt";
			break;
		}

		case 5:
		{
			name = "ID-Tasks-32DK-0DT.txt";
			break;
		}
	}
}

void setFileBenchmark_4producer(string &name, int benchmark)
{
	switch(benchmark)
	{
		case 1:
		{
			name = "realTasks_benchmark_1DK-3DT.txt";
			break;
		}

		case 2:
		{
			name = "realTasks_benchmark_3DK-1DT.txt";
			break;
		}

		case 3:
		{
			name = "realTasks_benchmark_2DK-2DT.txt";
			break;
		}

		case 4:
		{
			name = "realTasks_benchmark_4DK-0DT.txt";
			break;
		}

		case 5:
		{
			name = "realTasks_benchmark_0DK-4DT.txt";
			break;
		}
	}
}

void setFileBenchmark_6producer(string &name, int benchmark)
{
	switch(benchmark)
	{
		case 1:
		{
			name = "realTasks_benchmark_2DK-4DT.txt";
			break;
		}

		case 2:
		{
			name = "realTasks_benchmark_4DK-2DT.txt";
			break;
		}

		case 3:
		{
			name = "realTasks_benchmark_3DK-3DT.txt";
			break;
		}

		case 4:
		{
			name = "realTasks_benchmark_6DK-0DT.txt";
			break;
		}

		case 5:
		{
			name = "realTasks_benchmark_0DK-6DT.txt";
			break;
		}
	}
}

void setFileBenchmark_8producer(string &name, int benchmark)
{
	switch(benchmark)
	{
		case 1:
		{
			name = "realTasks_benchmark_3DK-5DT.txt";
			break;
		}

		case 2:
		{
			name = "realTasks_benchmark_5DK-3DT.txt";
			break;
		}

		case 3:
		{
			name = "realTasks_benchmark_4DK-4DT.txt";
			break;
		}

		case 4:
		{
			name = "realTasks_benchmark_8DK-0DT.txt";
			break;
		}

		case 5:
		{
			name = "realTasks_benchmark_0DK-8DT.txt";
			break;
		}
	}
}

void setFileBenchmark_16producer(string &name, int benchmark)
{
	switch(benchmark)
	{
		case 1:
		{
			name = "realTasks_benchmark_4DK-12DT.txt";
			break;
		}

		case 2:
		{
			name = "realTasks_benchmark_12DK-4DT.txt";
			break;
		}

		case 3:
		{
			name = "realTasks_benchmark_8DK-8DT.txt";
			break;
		}

		case 4:
		{
			name = "realTasks_benchmark_16DK-0DT.txt";
			break;
		}

		case 5:
		{
			name = "realTasks_benchmark_0DK-16DT.txt";
			break;
		}
	}
}

/**
 * @brief      Sets the file benchmark.
 * @details    This function calls to the functions which create the corresponding file name according to the paramenters.
 * @author     Antonio Jose Lazaro Munoz.
 * @date       15/05/2017
 * @param[in]     name       Reference to a string
 * @param[in]  benchmark  Benchmark id.
 * @param[in]  nproducer  Number of the producer threads.
 */
void setFileBenchmark(string &name, int benchmark, int nproducer)
{
	switch(nproducer)
	{
		case 4:
		{
			setFileBenchmark_4producer(name, benchmark);

			break;
		}

		case 8:
		{
			setFileBenchmark_8producer(name, benchmark);

			break;
		}

		case 6:
		{
			setFileBenchmark_6producer(name, benchmark);

			break;
		}

		case 16:
		{
			setFileBenchmark_16producer(name, benchmark);

			break;
		}
	}

}


/**
 * @brief      Proxy thread
 * @details    This function implements the proxy thread. This thread gets tasks from the producer buffers and inserts them
 * into a shared buffer of pending tasks. This buffer is called pending tasks buffer. This buffer is shared between 
 * the proxy thread and the handler gpu thread. The handler gpu thread is created by the proxy thread.
 * @author     Antonio Jose Lazaro Munoz.
 * @date       16/05/2017
 * 
 * @param[in]  gpu                     Gpu id
 * @param[in]  tid                     Thread id
 * @param[in]  producer_buffers        Producers buffers
 * @param[in]  nproducer               Number of producers thread
 * @param[in]  nepoch                  Number of epochs.
 * @param[in]  max_tam_batch           Maximum size of the batch
 * @param[in]  HTD_percentage          HtD percentage
 * @param[in]  computation_percentage  Computation percentage
 * @param[in]  DTH_percentage          DtH percentage
 * @param[in]  init_proxy              Synchronization variable
 * @param[in]  iter                    Iteration id.
 * @param[in]  nIter                   Number of iterations
 * @param[in]  elapsed_times           Execution times
 * @param[in]  n_launching_tasks       Number of launched tasks by producer thread.
 * @param[in]  scheduling_times        The scheduling times of the heuristic.
 */
void proxyThread(int gpu, int tid, vector<BufferTasks> &producer_buffers, int nproducer, int nepoch, int max_tam_batch, ifstream &fb, 
	atomic<int> &init_proxy, int iter, int nIter, float *elapsed_times, int *n_launching_tasks,
	vector<float>&scheduling_times, int *selected_order)
{
	//Boolean vector
	vector <bool> stop (nproducer);

	//Pending tasks buffer.
	BufferTasks pending_tasks_buffer;
  
  	for(int p = 0; p < nproducer; p++)
		stop[p] = false;
  
  	//Synchronization variable for handler gpu thread.
  	atomic<int> stop_handler_gpu;
  	stop_handler_gpu.store(0);

  	//Creamos un hilo planificador
  	thread scheduler_th(handler_gpu_func, gpu, ref(stop_handler_gpu), ref(pending_tasks_buffer), max_tam_batch, ref(fb),
  						nproducer, nepoch, 
						ref(init_proxy), iter, nIter, elapsed_times, n_launching_tasks,
						ref(scheduling_times), selected_order);

   	bool shutdown = false;

   	while(shutdown == false)
   	{
		//Loop the producer buffers.
		for(int p = 0; p < nproducer; p++)
		{
			//Available tasks to consume
			if(producer_buffers[p].getProducedElements() != 0)
			{
				infoTask task;

				//Get task from producer buffer.
				producer_buffers[p].getFront(task);
				//Consume task
				producer_buffers[p].popFront();

				//If the tasks is a STOP task
				if(task.id_task == STOP)
					stop[p] = true;
				else
				{
					//Insert task into the pending tasks buffer.
					pending_tasks_buffer.pushBack(task);
				}
			}
		}

		//Check if all producer threads have inserted STOP tasks and there are not pending tasks to consume.
		if(proxy_shutdown(stop, nproducer) == true && pending_tasks_buffer.getProducedElements() == 0)
		{
			stop_handler_gpu.store(1);
			shutdown = true;
		}
   	}

   	//Waiting the handler gpu thread.
   	scheduler_th.join();
}



void setFileIdTasks(string &name, int benchmark, int nproducer)
{
	switch(nproducer)
	{
		case 4:
		{
			setFileName_4producer(name, benchmark);

			break;
		}

		case 8:
		{
			setFileName_8producer(name, benchmark);

			break;
		}

		case 6:
		{
			setFileName_6producer(name, benchmark);

			break;
		}

		case 16:
		{
			setFileName_16producer(name, benchmark);

			break;
		}
	}
}

/**
 * @brief      Main function,
 * @details    Main function.
 * @author 	   Bernabe Lopez Albelda.
 * @date       07/03/2018
 * @param[in]  argc  Number of input arguments.
 * @param      argv  Input arguments.
 *
 * @return     
 */
int main(int argc, char *argv[])
{
	if(argc != 13)
	{
	  cout << "Execute: <program> <gpu> <nproducer> <nepoch> <max_tam_batch> <task_file_path> <time_file_path>"; 
	  cout << " <benchmark> <ditribution_type> <interval1> <interval2> <nIter> <heuristic>" << endl;
	  exit(EXIT_SUCCESS);
	}

	int gpu           = atoi(argv[1]);		//GPU id
	int nproducer     = atoi(argv[2]);		//Number of producer threads.
	int nepoch        = atoi(argv[3]);		//Number of epochs.
	int max_tam_batch = atoi(argv[4]);		//Max size of the batch.
	int benchmark     = atoi(argv[7]);		//Benchmark id
	int nIter         = atoi(argv[11]);		//Number of iterations

	string str_task_file_path (argv[5]);	//Directory Path of the tasks configuration file
	string str_time_file_path(argv[6]);		//Directory Path of the inteval times file
	string str_type_distribution(argv[8]);	//Type of the time distribution
	string str_interval1(argv[9]);			//Minimum time interval
	string str_interval2(argv[10]);			//Maximum time interval
	string str_heuristic(argv[12]);			//Heuristic used
	
	int *h_order = new int [N_TASKS * nepoch];
	
	//get name server
	char hostname[50];
	gethostname(hostname, 50);
	
	//get name gpu
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpu);
	
	for(int epoch = 0; epoch < nepoch; epoch++)
	{
		string name_orderTasks_file = "Order-Tasks_bench" + to_string(benchmark) + "_uniform_"
									+ to_string(epoch+1) + "e_"
									+ to_string(N_TASKS) + "p_i0-0-" + hostname + "-"
									+ prop.name + "-" + str_heuristic + ".txt";
		ifstream order_file(name_orderTasks_file);
		
		string linea;
		getline(order_file, linea);
		vector<string> v(tokenize (linea, " "));
		
		for(int app = 0; app < N_TASKS; app++)
			h_order[epoch * N_TASKS + app] = atoi(v[app].c_str());
			
		order_file.close();
	}
	
	///We construct the path of the file name of the intervals times
	string name_times_file = str_time_file_path + "times_" + str_type_distribution 
							+ "_" + to_string(nproducer) + "p_40e_i"
							+ str_interval1 + "-" + str_interval2 + ".txt";

	string name;
	//Set path of the benchmark file name.
	string name_real_benchmark;
	setFileIdTasks(name, benchmark, nproducer);
	setFileBenchmark(name_real_benchmark, benchmark, nproducer);
	string name_task_file = str_task_file_path + name;
	string real_benchmark_file = str_task_file_path + name_real_benchmark;
	
	//Select GPU
  	cudaSetDevice(gpu);

	
  	int * id_launched_tasks = new int[nproducer*nepoch];	//id of the launched tasks by the producer threads
  	float *waiting_times    = new float[nproducer*nepoch];		//Waiting time for the tasks insertions for the producer threads

  	float *elapsed_times = new float[nIter];	//Execution times
	
	//We read the tasks id file and the waiting times file.
	//Open files.
	
	ifstream fe(name_task_file);
	ifstream ft(name_times_file);
	cout << real_benchmark_file << endl;
	ifstream fb(real_benchmark_file);

	for(int p = 0; p < nproducer; p++)
	{
		string line;
		string line_times;

		getline(fe, line);
		getline(ft, line_times);
		
		vector<string> v(tokenize(line, "\t"));

		vector<string> v_times(tokenize(line_times, "\t"));
		
		for(int i = 0; i < nepoch; i++)
		{
			id_launched_tasks[p*nepoch + i] = atoi(v[i].c_str());
			waiting_times[p*nepoch + i] = atof(v_times[i].c_str());
		}
	}
	
	fe.close();
	ft.close();

	//Creating a shared buffers between producers and proxy
	vector <BufferTasks> producer_buffers(nproducer);
	atomic<int> init_proxy;			//Variable of synchronization for the proxy.
	atomic<int> sync_producers;		//Variable of synchronization for the producers. //NO SE UTILIZA
	vector<float>scheduling_times;	//Scheduling times of the heuristic
	
	int *n_launching_tasks = new int[max_tam_batch];			//Number of launched tasks by thread
	memset(n_launching_tasks, 0, max_tam_batch*sizeof(int));

	//Launching producer
	for(int iter = 0; iter < nIter; iter++)
	{
		cerr << "Heur. Benchmark: " << benchmark << " - Iter: " << iter << endl;
		memset(n_launching_tasks, 0, max_tam_batch*sizeof(int));

		//Launching proxy thread

		//Creating producer
	    thread *producer_vector = new thread[nproducer];

		init_proxy.store(0);
		sync_producers.store(0);

		thread proxy(proxyThread, gpu, nproducer + 1, ref(producer_buffers), nproducer, nepoch, max_tam_batch, ref(fb),
		ref(init_proxy), iter, nIter, elapsed_times, 
		n_launching_tasks, ref(scheduling_times), h_order);

		while(init_proxy.load() == 0);

		for(int tid = 0; tid < nproducer; tid++)
			producer_vector[tid] = thread(producerThread, gpu, tid, ref(producer_buffers), nepoch, id_launched_tasks, 
								waiting_times, ref(sync_producers), nproducer);

		for(int tid = 0; tid < nproducer; tid++)
			producer_vector[tid].join();

		//Waiting Proxy thread
		proxy.join();

		delete [] producer_vector;
	}
	
	string name_fich_time_rep = "rep_times_benchmark_" + to_string(benchmark) + "_" + to_string(nproducer) + "p_" + to_string(nepoch) 
							+ "e_i" + str_interval1 + "-" + str_interval2 + "-" + str_heuristic + ".txt";
	
	ofstream f_time_rep(name_fich_time_rep);

	for(int i = 0; i < nIter; i++)
	{
		f_time_rep << elapsed_times[i] << endl;
	}

	f_time_rep.close();
						
	string name_fich_results = "results_benchmark_" + to_string(benchmark) + "_" + to_string(nproducer) + "p_" + to_string(nepoch) 
							+ "e_i" + str_interval1 + "-" + str_interval2 + "-" + str_heuristic + ".txt";
	ofstream fresult(name_fich_results);

	fresult << getMedianTimeG(elapsed_times, nIter) << endl;
	fresult.close();

	delete [] id_launched_tasks;
	delete [] waiting_times;
	delete [] elapsed_times;
	delete [] n_launching_tasks;
	
	delete [] h_order;

	return 0;
}
