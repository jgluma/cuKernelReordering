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
//#include "sintetic_benchmarks.h"
//#include "benchmarks4tasks/sinteticBenchmarks-4tasks.h"
//#include "benchmarks6tasks/sinteticBenchmarks-6tasks.h"
//#include "benchmarks8tasks/sinteticBenchmarks-8tasks.h"
//#include "benchmarks16tasks/sinteticBenchmarks-16tasks.h"
//#include "benchmarks32tasks/sinteticBenchmarks-32tasks.h"
//#include "sinteticTask.h"
#include "TaskTemporizer.h"
#include <sys/time.h>
#include <unistd.h> 
#include <chrono>

#include <limits.h>

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
//#include "cublasrotm.h"
//#include "FFT.h"
#include "BlackScholes.h"
#include "FastWalshTransform.h"
#include "ConvolutionSeparable.h"
#include "Needle.h"
#include "Gaussian.h"
#include "ParticleFilter.h"
#include "PathFinder.h"
#include "SobolQRNG.h"

#define INF_TIME	1000000000

int order_tasks[12] = {4, 0, 3, 1, 5, 2, 3, 1, 0, 5, 4, 2};

using namespace std;

#define PRINT_SIMULATOR_TRACE	0
#define TASKS_TIMES_TO_FILE	1

#define LAUNCH_LAST_HTD	1
#define LAUNCH_BATCH	0

#define DK	0
#define DT	1

#define ORDER_TO_FILE	0

#define PRUEBA_KDTH_MIN 0

//#define MAX_N_EPOCH	40

struct infoCommand{
	int id_stream;				//ID stream
	int id_epoch;				//ID epoca
	int id;						//ID del comando en la simulacion actual.
	
	float t_ini;				//Tiempo de inicio del comando
	float t_fin;				//Tiempo de fin del comando
	float t_estimated_fin;		//Tiempo de fin estimado para el comando
	float t_CPU_GPU;			//Duracion comando HTD
	float t_GPU_CPU;			//Duracion comando DTH
	float t_kernel;				//Duracion comando Kernel
	float t_overlap_CPU_GPU;	//Tiempo de duracion del comando HTD cuando esta solapado.
	float t_overlap_GPU_CPU;	//Tiempo de duracion del comando DTH cuando esta solapado.
	bool active_htd;			//Flag que indica que el comando tiene una dependencia con un comando HTD.
	bool ready;					//Flag para indicar que el comando esta listo
	bool enqueue;				//Flag que indica que el comando debe ser encolado para tener en cuenta en una epoca posterior.
	bool overlapped;			//Flag que indica que el comando de transferencia esta siendo solapado con otro comando de transferencia. 
	bool launched;				//Flag que indica que el comando ya ha sido simulado en una epoca anterior.
	
	std::deque<infoCommand>::iterator next_command;	//Puntero a otro comando debido a que contiene una dependencia.
};

//Colas de comandos para el simulador
deque<infoCommand>deque_simulation_DTH;
deque<infoCommand>deque_simulation_HTD;
deque<infoCommand>deque_simulation_K;

//Colas de comandos a tener en cuenta para una epoca posterior.
deque<infoCommand>deque_current_DTH;
deque<infoCommand>deque_current_HTD;
deque<infoCommand>deque_current_K;

deque<infoCommand>deque_execution_DTH;
deque<infoCommand>deque_execution_HTD;
deque<infoCommand>deque_execution_K;

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

/* esto seguro que sabes lo que hace ... */
void swap(int *x, int *y) { 
  int temp = *x;
  *x = *y;
  *y = temp;
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

			/*case MYFFT:
			{

				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new FFT(params[0]));

				delete [] params;

				break;
			}*/

			/*case ROTM:
			{

				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				tasks_v.push_back(new CUBLASROTM(params[0]));

				delete [] params;

				break;
			}*/

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

void executeKernels(int gpu, int nIter, float *time_kernels, ifstream &fb, int n_tasks)
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

     for(int t = 0; t < n_tasks; t++){
     	string line;
		
		getline(fb, line);
		
		vector<string> v(tokenize(line, "\t"));

		int type_task = atoi(v[0].c_str());
		int num_param = atoi(v[1].c_str());

		Task *task = NULL;
		
		switch(type_task)
		{
			case MM:
			{	
				int *sizes = new int[num_param];

				for(int i = 0; i < num_param; i++)
					sizes[i] = atoi(v[i+2].c_str());
				
				//Creamos la tarea
				task = new MatrixMult(sizes);

				delete [] sizes;

				break;
			}

			case HST:
			{
				int nframes = atoi(v[2].c_str());

				task = new Histogram(nframes);

				break;
			}

			case CONV:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new ConvolutionSeparable(params[0], params[1], params[2]);

				delete [] params;

				break;
			}

			case TM:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new Transpose(params[0], params[1]);

				delete [] params;
				break;
			}

			case FWT:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new FastWalshTransform(params[0], params[1]);

				delete [] params;
				break;
			}

			case BS:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new BlackScholes(params[0], params[1]);

				delete [] params;

				break;
			}

			case GS:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new Gaussian(params[0]);

				delete [] params;

				break;
			}

			case ND:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new Needle(params[0], params[1], params[2]);

				delete [] params;

				break;
			}

			case PAF:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new ParticleFilter(params[0], params[1], params[2], params[3]);

				delete [] params;

				break;
			}

			case PF:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new PathFinder(params[0], params[1], params[2]);

				delete [] params;

				break;
			}

			case VA:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new VectorADD(params[0]);

				delete [] params;

				break;
			}

			/*
			case MYFFT:
			{

				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new FFT(params[0]);

				delete [] params;

				break;
			}
			*/

			/*
			case ROTM:
			{

				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new CUBLASROTM(params[0]);

				delete [] params;

				break;
			}
			*/
			case SB:
			{
				int *params = new int[num_param];

				for(int i = 0; i < num_param; i++)
					params[i] = atoi(v[i+2].c_str());

				task = new SobolQRNG(params[0], params[1], gpu);

				delete [] params;

				break;
			}
		}
		
		//Alloc host Memory
  		task->allocHostMemory();
  		//Alloc device Memory
  		task->allocDeviceMemory();

  		//Generating input data
  		task->generatingData();

  		float *median_time = new float [nIter]; 
		float time = 0;

		for(int k = 0; k < nIter; k++)
		{
			task->memHostToDeviceAsync(stream);
			cudaEventRecord(start_event, 0);
			task->launch_kernel_Async(stream);
			cudaEventRecord(stop_event, 0);
			cudaEventSynchronize(stop_event);
			float t = 0;
			cudaEventElapsedTime(&t, start_event, stop_event);
			median_time[k] = t;
		}

		//Free host memory
  		task->freeHostMemory();
  		//Free device memory
  		task->freeDeviceMemory();

  		time = getMedianTimeG(median_time, nIter);

  		time_kernels[t] = time;
	
		delete [] median_time;
		delete task;
    }
	 
    cudaStreamDestroy(stream);
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
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

		/*case 32:
		{

			setFileName_32producer(name, benchmark);

			break;
		}*/
	}
}

/**
 * @brief      Main function,
 * @details    Main function.
 * @author 	   Bernabe Lopez Albelda.
 * @date       27/02/2018
 * @param[in]  argc  Number of input arguments.
 * @param      argv  Input arguments.
 *
 * @return     
 */
int main(int argc, char *argv[])
{
	if(argc != 12)
	{
	  cout << "Execute: <program> <gpu> <nproducer> <nepoch> <max_tam_batch> <task_file_path>"; 
	  cout << " <time_file_path> <benchmark> <ditribution_type> <interval1> <interval2> <nIter>" << endl;
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
	
	///We construct the path of the file name of the intervals times
	string name_times_file = str_time_file_path + "times_" + str_type_distribution 
							+ "_" + to_string(nproducer) + "p_" + to_string(nepoch) 
							+ "e_i" + str_interval1 + "-" + str_interval2 + ".txt";

	string name;
	//Set path of the benchmark file name.
	string name_real_benchmark;
	setFileIdTasks(name, benchmark, nproducer);
	setFileBenchmark(name_real_benchmark, benchmark, nproducer);
	string name_task_file = str_task_file_path + name;
	string real_benchmark_file = str_task_file_path + name_real_benchmark;
	
	//get name server
	char hostname[50];
	gethostname(hostname, 50);
	
	//get name gpu
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);

	string name_matrixTasks_file = "Matrix-Tasks_bench" + to_string(benchmark) + "_" + 
											str_type_distribution 
											+ "_" + to_string(nproducer) + "p_" + to_string(nepoch) 
											+ "e_i" + str_interval1 + "-" + str_interval2 + "-" + hostname + "-" + prop.name + "-HEURISTICO.txt";

	ofstream fich_tasks_matrix(name_matrixTasks_file);
	
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

	int *selected_order = new int[N_TASKS*nepoch]; 

	//Launching producer
		
	//cerr << "Heur. Benchmark: " << benchmark << " - Iter: " << iter << endl;
	memset(n_launching_tasks, 0, max_tam_batch*sizeof(int));

	//Launching proxy thread

	//Creating producer
	thread *producer_vector = new thread[nproducer];

	init_proxy.store(0);
	sync_producers.store(0);	//NO SE UTILIZA

	//Scheduling batch.
	Batch scheduling_batch(max_tam_batch);
	//Execution batch.
	Batch execution_batch(max_tam_batch);

	int scheduled_tasks = 0;
	int nstreams = nproducer;

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
		

	float *time_kernels = new float[N_TASKS];


	executeKernels(gpu, nIter, time_kernels, fb, nstreams);
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

  	//Transfers times of the tasks
  	float *estimated_time_HTD                    = new float[N_TASKS]; 
  	float *estimated_time_DTH                    = new float[N_TASKS]; 
  	float *estimated_overlapped_time_HTD         = new float[N_TASKS]; 
 	float *estimated_overlapped_time_DTH         = new float[N_TASKS];
  
  	//Transfers times of the tasks according to the order of the scheduling batch 
  	float *estimated_time_HTD_per_stream_execute            = new float[nstreams]; 
  	float *estimated_time_DTH_per_stream_execute            = new float[nstreams]; 
  	float *estimated_overlapped_time_HTD_per_stream_execute = new float[nstreams]; 
  	float *estimated_overlapped_time_DTH_per_stream_execute = new float[nstreams];
  	int *bytes_htd = new int[N_TASKS];
  	int *bytes_dth = new int[N_TASKS];


  	float LoHTD          = 0.0;
  	float LoDTH          = 0.0;
  	float GHTD           = 0.0;
  	float overlappedGHTD = 0.0;
  	float GDTH           = 0.0;
  	float overlappedGDTH = 0.0;
    
  	//PCIe microbencmarking
  	microbenchmarkingPCI(gpu, &LoHTD, &LoDTH, &GHTD, &overlappedGHTD, &GDTH, &overlappedGDTH, nIter);

  	for(int app = 0; app < N_TASKS; app++)
  	{
   		tasks_v.at(app)->getBytesHTD(&bytes_htd[app]);
  		tasks_v.at(app)->getBytesDTH(&bytes_dth[app]);

   		tasks_v.at(app)->getTimeEstimations_HTD_DTH(gpu, &estimated_time_HTD[app],
   			&estimated_time_DTH[app],
			&estimated_overlapped_time_HTD[app], 
			&estimated_overlapped_time_DTH[app], 
			LoHTD, LoDTH, GHTD, GDTH, overlappedGHTD, overlappedGDTH);
   	}
  
   	float *h_time_kernels_rep = new float[nIter * N_TASKS]; 
  	float *h_time_kernels_tasks = new float[N_TASKS]; 
  
  	for(int i = 0; i < N_TASKS; i++)
		h_time_kernels_tasks[i] = time_kernels[i];
	
	delete [] time_kernels;

	for(int app = 0; app < N_TASKS; app++)
	{
			fich_tasks_matrix << "Tarea " << app << ":\t"
							  << estimated_time_HTD[app] << "\t" 
							  << h_time_kernels_tasks[app] << "\t"
							  << estimated_time_DTH[app] << "\t"
							  << estimated_overlapped_time_HTD[app] << "\t"
							  << estimated_overlapped_time_DTH[app] << "\t"
							  << bytes_htd[app] << "\t"
							  << bytes_dth[app] << "\t";

			if((estimated_time_HTD[app] + estimated_time_DTH[app]) > h_time_kernels_tasks[app])
				fich_tasks_matrix << "DT" << endl;
			else
				if((estimated_time_HTD[app] + estimated_time_DTH[app]) < h_time_kernels_tasks[app])
					fich_tasks_matrix << "DK" << endl;
				else
					fich_tasks_matrix << "DT-DK" << endl;
	}

	delete [] bytes_dth;
	delete [] bytes_htd;

	delete [] id_launched_tasks;
	delete [] waiting_times;
	delete [] elapsed_times;
	delete [] n_launching_tasks;

	return 0;
}
