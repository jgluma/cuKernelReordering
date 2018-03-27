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

#include <limits.h>

#define INF_TIME	1000000000

using namespace std;

#define PRINT_SIMULATOR_TRACE	0
#define PRINT_NEH_TRACE 1

#define DK	0
#define DT	1

#define PRUEBA_KDTH_MIN 0

#define MEASURE_TIME 0

struct infoCommand{
	
	int id_stream;				//ID stream
	int id_epoch;				//ID epoca
	int id;						//ID del comando en la simulacion actual.
	int id_chunk;
	int id_task;
	
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
	float t_fin_unfair;
	float t_estimated_dur;
	
	std::deque<infoCommand>::iterator next_command;	//Puntero a otro comando debido a que contiene una dependencia.
	std::deque<infoCommand>::iterator next_command_2;
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

float getMinTime(float *h_times, int N)
{
	float min = h_times[0];
	
	for(int i = 1; i < N; i++)
	{
		if(min > h_times[i])
			min = h_times[i];
	}
	
	return min;
}

int getMaxTask(float *h_times, int *inserted_tasks, int N)
{
	float max = -1;
	
	if(inserted_tasks[0] != 1)
		max = h_times[0];
		
	int index = 0;
	for(int i = 1; i < N; i++)
	{
		if(max < h_times[i] && inserted_tasks[i] != 1)
		{
			max = h_times[i];
			index = i;
		}
	}
	
	return index;
}

int getNumberTypeDominant(int *h_type_dominant, int N, int type_dominant)
{
	int num = 0;
	
	for(int t = 0; t < N; t++)
	{
		if(h_type_dominant[t] == type_dominant)
			num++;
	}
	
	return num;
}

float simulator2CopyEngine_bubble_command_v6_FAIR(float *time_kernels, int chunks_number, float *time_CPU_GPU, 
						   float *time_GPU_CPU, float *overlapped_time_CPU_GPU, 
						   float *overlapped_time_GPU_CPU, int *permutation, int *execute_batch, int n_app, 
						   float t_previous_ini_htd, float t_previous_ini_kernel, float t_previous_ini_dth,
						   float t_previous_fin_htd, float t_previous_fin_kernel, float t_previous_fin_dth,
						   float t_previous_overlap_htd, float t_previous_overlap_dth,
						   float *t_current_ini_htd, float *t_current_ini_kernel, float *t_current_ini_dth,
						   float *t_current_fin_htd, float *t_current_fin_kernel, float *t_current_fin_dth,
						   float *t_current_overlap_htd, float *t_current_overlap_dth,
						   float *t_previous_last_dth_stream, float *t_current_last_dth_stream, int id_epoch)
{
	
	
	
	
	//for(int i = 0; i < nstreams; i++)
	//	cout << "K " << i << " Tiempo " << time_kernels[i] << endl;
	int nstreams = n_app;
	
	infoCommand kernel_command;
	infoCommand HTD_command;
	infoCommand DTH_command;
	float time_counter = 0;
	
	float last_command = 0;
	
	//Limpiamos las colas de simulacion
	deque_simulation_HTD.clear();
	deque_simulation_K.clear();
	deque_simulation_DTH.clear();
	//Limpiamos las colas CURRENT
	deque_current_HTD.clear();
	deque_current_K.clear();
	deque_current_DTH.clear();
	
	//Insertamos en las colas de simulacion los comandos de las colas de ejecucion que pertenecen a la epoca anterior.
	for(deque<infoCommand>::iterator current_HTD = deque_execution_HTD.begin(); current_HTD != deque_execution_HTD.end(); current_HTD++)
	{
		HTD_command.id_stream = current_HTD->id_stream;
		HTD_command.id_epoch = current_HTD->id_epoch;
		HTD_command.id = current_HTD->id;	
		HTD_command.ready = true;
		HTD_command.overlapped = false;
		HTD_command.enqueue = false;
		HTD_command.launched = true;
		HTD_command.t_ini = current_HTD->t_ini;
		HTD_command.t_fin = current_HTD->t_fin;
		HTD_command.t_fin_unfair = current_HTD->t_fin_unfair;
		HTD_command.t_CPU_GPU = current_HTD->t_CPU_GPU;
		HTD_command.t_overlap_CPU_GPU = current_HTD->t_overlap_CPU_GPU;
		
		deque_simulation_HTD.push_back(HTD_command);
		
	}
	
	for(deque<infoCommand>::iterator current_K = deque_execution_K.begin(); current_K != deque_execution_K.end(); current_K++)
	{
		
		kernel_command.id_stream = current_K->id_stream;
		kernel_command.id = current_K->id;
		kernel_command.id_epoch = current_K->id_epoch;
		kernel_command.ready = false;
		kernel_command.enqueue = false;
		kernel_command.launched = true;
		kernel_command.t_ini = current_K->t_ini;
		kernel_command.t_fin = current_K->t_fin;
		kernel_command.t_kernel = current_K->t_kernel;
		
		deque_simulation_K.push_back(kernel_command);
		
	}
	
	for(deque<infoCommand>::iterator current_DTH = deque_execution_DTH.begin(); current_DTH != deque_execution_DTH.end(); current_DTH++)
	{
		
		DTH_command.id_stream = current_DTH->id_stream;
		DTH_command.id = current_DTH->id;
		DTH_command.id_task = current_DTH->id_task;
		DTH_command.id_epoch = current_DTH->id_epoch;
		DTH_command.ready = false;
		DTH_command.overlapped = false;
		DTH_command.enqueue = false;
		DTH_command.launched = true;
		DTH_command.active_htd = false;
		DTH_command.t_ini = current_DTH->t_ini;
		DTH_command.t_fin = current_DTH->t_fin;
		DTH_command.t_fin_unfair = current_DTH->t_fin_unfair;
		DTH_command.t_GPU_CPU = current_DTH->t_GPU_CPU;
		DTH_command.t_overlap_GPU_CPU = current_DTH->t_overlap_GPU_CPU;
		
		deque_simulation_DTH.push_back(DTH_command);
		
	}
		
	
	
	int num_HTD_enqueue = deque_simulation_HTD.size();
	int num_DTH_enqueue = deque_simulation_DTH.size();
	int num_K_enqueue = deque_simulation_K.size();
	
	
	//Establecemos las dependencias entre los comandos de las epocas anteriores
	//HTD->K
	
	for(deque<infoCommand>::iterator current_K = deque_simulation_K.begin(); current_K != deque_simulation_K.end(); current_K++)
	{
		
		current_K->ready = true;
		for(deque<infoCommand>::iterator current_HTD = deque_simulation_HTD.begin(); 
			current_HTD != deque_simulation_HTD.end(); current_HTD++)
		{
			if(current_HTD->id_stream == current_K->id_stream && current_HTD->id_epoch == current_K->id_epoch)
			{				
				current_K->ready = false;
				current_HTD->next_command = current_K;
				
			}
			
			
		}
		
		
		
	}
	
	//K->DTH
	
	for(deque<infoCommand>::iterator current_DTH = deque_simulation_DTH.begin(); current_DTH != deque_simulation_DTH.end(); current_DTH++)
	{
		
		current_DTH->ready = true;
		for(deque<infoCommand>::iterator current_K = deque_simulation_K.begin(); 
			current_K != deque_simulation_K.end(); current_K++)
		{
			if(current_K->id_stream == current_DTH->id_stream && current_K->id_epoch == current_DTH->id_epoch)
			{
				current_DTH->ready = false;
				current_K->next_command = current_DTH;
				
			}
			
			
		}
		
		
		
	}
	
	int num_tasks = 0;
	//Introducimos en las colas de simulacion los nuevos comandos
	for(int i = 0; i < nstreams; i++)
	{
				
			
			//HTD_command.id_stream = i;
			HTD_command.id = i;
			HTD_command.id_stream = execute_batch[permutation[i]];
			HTD_command.id_epoch = id_epoch;
			HTD_command.ready = false;
			HTD_command.overlapped = false;
			HTD_command.t_ini = INF_TIME;
			HTD_command.t_fin = INF_TIME;
			HTD_command.t_CPU_GPU =   time_CPU_GPU[permutation[i]]; //time_CPU_GPU[execute_batch[permutation[i]]];  time_CPU_GPU[i];
			HTD_command.enqueue = false;
			HTD_command.launched = false;
			HTD_command.t_overlap_CPU_GPU =  overlapped_time_CPU_GPU[permutation[i]]; //overlapped_time_CPU_GPU[execute_batch[permutation[i]]]; overlapped_time_CPU_GPU[i];
		
			
			//kernel_command.id_stream = i;
			kernel_command.id = i;
			kernel_command.id_stream = execute_batch[permutation[i]];
			kernel_command.id_epoch = id_epoch;
			kernel_command.ready = false;
			kernel_command.t_ini = INF_TIME;
			kernel_command.t_fin = INF_TIME;
			kernel_command.t_kernel =    time_kernels[permutation[i]]; //time_kernels[execute_batch[permutation[i]]];  time_kernels[i];
			kernel_command.enqueue = false;
			kernel_command.launched = false;
			
			
			
			//DTH_command.id_stream = i;
			DTH_command.id = i;
			DTH_command.id_stream = execute_batch[permutation[i]];
			DTH_command.id_epoch = id_epoch;
			DTH_command.id_task = execute_batch[permutation[i]];
			DTH_command.ready = false;
			DTH_command.overlapped = false;
			DTH_command.enqueue = false;
			DTH_command.active_htd = false;
			DTH_command.t_ini = INF_TIME;
			DTH_command.t_fin = INF_TIME;
			DTH_command.t_GPU_CPU =   time_GPU_CPU[permutation[i]]; //time_GPU_CPU[execute_batch[permutation[i]]]; time_GPU_CPU[i];
			DTH_command.t_overlap_GPU_CPU = overlapped_time_GPU_CPU[permutation[i]];   //overlapped_time_GPU_CPU[execute_batch[permutation[i]]];  overlapped_time_GPU_CPU[i];
			DTH_command.launched = false;
			
			
			
			deque_simulation_HTD.push_back(HTD_command);
				
			deque_simulation_K.push_back(kernel_command);
				
			deque_simulation_DTH.push_back(DTH_command);
				
			num_tasks++;
			
	}
	
	
	//Declaramos unos punteros la inicio de los nuevos comandos en las colas.
	std::deque<infoCommand>::iterator current_K_begin = deque_simulation_K.begin() + num_K_enqueue;
	std::deque<infoCommand>::iterator current_HTD_begin = deque_simulation_HTD.begin() + num_HTD_enqueue;
	std::deque<infoCommand>::iterator current_DTH_begin = deque_simulation_DTH.begin() + num_DTH_enqueue;
	//Declaramos unos punteros para recorrer los nuevos comandos en las colas
	std::deque<infoCommand>::iterator current_K = deque_simulation_K.begin() + num_K_enqueue;
	std::deque<infoCommand>::iterator current_HTD = deque_simulation_HTD.begin() + num_HTD_enqueue;
	std::deque<infoCommand>::iterator current_DTH = deque_simulation_DTH.begin() + num_DTH_enqueue;
	
	
	//Establecemos las dependencias entre los nuevos comandos.
	//Inicializamos los primeros tiempos de inicio de las colas
	
	/*for(int i = 0; i < nstreams; i++)
	{
		current_HTD->ready = true;
		current_HTD++;
		
	}*/
	
	
	
	for(int i = 0; i < nstreams; i++)
	{
		current_HTD->ready = true;
		
		for(deque<infoCommand>::iterator current_DTH = deque_simulation_DTH.begin(); 
			current_DTH != current_DTH_begin; current_DTH++)
		{
			if(current_DTH->id_stream == current_HTD->id_stream)
			{
				//Si existe un DTH perteneciente al stream del HTD
				current_HTD->ready = false;
				current_DTH->next_command = current_HTD;
				current_DTH->active_htd = true;
				
			}
		
		
		}
		
		current_HTD++;
		
	}
	
	
	
	//HTD->K
	
	for(current_HTD = current_HTD_begin; current_HTD != deque_simulation_HTD.end(); current_HTD++)
	{
		current_HTD->next_command = current_K;
		current_K++;
		
	}
	
	//K->DTH
	current_DTH = current_DTH_begin;

	for(current_K = current_K_begin; current_K != deque_simulation_K.end(); current_K++)
	{
		
		current_K->next_command = current_DTH;
		current_DTH++;
	}
	
	
	
	//Si NO hay ningun comando en las colas de ejecucion inicializamos time_counter a 0
	//de lo contrario lo inicializaremos al tiempo de inicio menor entre los comandos de las
	//colas de ejecucion
	if(deque_execution_HTD.size() == 0 
	&& deque_execution_K.size() == 0 
	&& deque_execution_DTH.size() == 0)
		time_counter = 0;
	else
	{
		if(deque_execution_HTD.size() != 0)
			time_counter = deque_execution_HTD.begin()->t_ini;
		
		if(time_counter > deque_execution_K.begin()->t_ini && deque_execution_K.size() != 0)
			time_counter = deque_execution_K.begin()->t_ini;
		
		if(time_counter > deque_execution_DTH.begin()->t_ini && deque_execution_DTH.size() != 0)
			time_counter = deque_execution_DTH.begin()->t_ini;
		
	}
	
	
	float *time_queues = (float *)malloc(3*sizeof(float));
	
	int i = 0;
	
	while(!deque_simulation_HTD.empty() || !deque_simulation_K.empty() || !deque_simulation_DTH.empty())
	{
		if(!deque_simulation_HTD.empty())
			current_HTD = deque_simulation_HTD.begin();
		if(!deque_simulation_K.empty())
			current_K = deque_simulation_K.begin();
		if(!deque_simulation_DTH.empty())
			current_DTH = deque_simulation_DTH.begin();
		if(current_DTH->id_task < 0 || current_DTH->id_task > N_TASKS - 1) // BERNABE
			current_DTH->id_task = current_DTH->id_task % N_TASKS;
		
#if PRINT_SIMULATOR_TRACE			
		printf("\n******************** ITERACION %d ****************\n", i);
			
		printf("Estado Colas...\n");
#endif
		
		
		//Si es un comando de la epoca actual, inicializamos sus tiempos de inicio y fin
		if(!deque_simulation_HTD.empty() && current_HTD->t_ini == INF_TIME && current_HTD->ready == true)
		{
			current_HTD->t_ini = time_counter;
			
			//if(t_previous_last_dth_stream[current_HTD->id_stream] > time_counter)
				//current_HTD->t_ini = t_previous_last_dth_stream[current_HTD->id_stream];
				
			current_HTD->t_fin = current_HTD->t_ini + current_HTD->t_CPU_GPU;
			current_HTD->t_fin_unfair = current_HTD->t_ini + current_HTD->t_CPU_GPU;
			
			
		}
		
#if PRINT_SIMULATOR_TRACE
		printf("HTD t_ini %f t_fin %f Duration %f\n", current_HTD->t_ini, current_HTD->t_fin, current_HTD->t_fin - current_HTD->t_ini);
#endif

		if(!deque_simulation_K.empty() && current_K->t_ini == INF_TIME && current_K->ready == true)
		{
			current_K->t_ini = time_counter;
			current_K->t_fin = current_K->t_ini + current_K->t_kernel;
			
			
		}
		
#if PRINT_SIMULATOR_TRACE
		printf("K t_ini %f t_fin %f Duration %f\n", current_K->t_ini, current_K->t_fin, current_K->t_fin - current_K->t_ini);
#endif

		if(!deque_simulation_DTH.empty() && current_DTH->t_ini == INF_TIME && current_DTH->ready == true)
		{
			current_DTH->t_ini = time_counter;
			current_DTH->t_fin = current_DTH->t_ini + current_DTH->t_GPU_CPU;
			current_DTH->t_fin_unfair = current_DTH->t_ini + current_DTH->t_GPU_CPU;
			
			
		}

#if PRINT_SIMULATOR_TRACE
		printf("DTH t_ini %f t_fin %f Duration %f\n", current_DTH->t_ini, current_DTH->t_fin, current_DTH->t_fin - current_DTH->t_ini);
#endif		
		
		//SOLAPAMIENTOS ENTRE TRANSFERENCIAS
		
		
		if(!deque_simulation_HTD.empty() && !deque_simulation_DTH.empty()
		&& current_HTD->t_ini != INF_TIME && current_DTH->t_ini != INF_TIME
		&& current_HTD->overlapped == false && current_DTH->overlapped == false
		&& current_HTD->ready == true && current_DTH->ready == true
		&& (current_HTD->launched == false || current_DTH->launched == false))
		{
			
			//Version 2 solapamiento
			float time_overlap_CPU_GPU;
			float time_overlap_GPU_CPU;
			
			float t_CPU_GPU = current_HTD->t_CPU_GPU;
			float t_overlap_CPU_GPU = current_HTD->t_overlap_CPU_GPU;
			float t_GPU_CPU = current_DTH->t_GPU_CPU;
			float t_overlap_GPU_CPU = current_DTH->t_overlap_GPU_CPU;
			
			
			//Solapamiento HTD version 2
			if(current_HTD->t_ini == time_counter)
			{
				
				
				if(current_DTH->t_fin < current_HTD->t_fin)	//Solapamos solo una parte de la transferencia HTD.
				{
								
								
					
					time_overlap_CPU_GPU = ((current_DTH->t_fin - time_counter)*t_overlap_CPU_GPU)/t_CPU_GPU;
							
					time_overlap_CPU_GPU = time_overlap_CPU_GPU + (current_HTD->t_fin - current_DTH->t_fin);
								
				}
				else
				{	
							
					
					//time_overlap_CPU_GPU = overlapped_time_CPU_GPU;
					time_overlap_CPU_GPU = t_overlap_CPU_GPU;
				}
						
						
			}
			else
			{
						
				if(current_DTH->t_fin < current_HTD->t_fin) //Solapamos solo una parte de la transferencia HTD.
				{
						
						time_overlap_CPU_GPU = ((current_DTH->t_fin - time_counter)*t_overlap_CPU_GPU)/t_CPU_GPU;
							
						time_overlap_CPU_GPU = (time_counter - current_HTD->t_ini) + time_overlap_CPU_GPU + (current_HTD->t_fin - current_DTH->t_fin);
							
							
				}
				else
				{
								
							
						//time_overlap_CPU_GPU = ((current_HTD->t_fin - time_counter)*overlapped_time_CPU_GPU)/time_CPU_GPU;
						time_overlap_CPU_GPU = ((current_HTD->t_fin - time_counter)*t_overlap_CPU_GPU)/t_CPU_GPU;
							
						time_overlap_CPU_GPU  = (time_counter - current_HTD->t_ini) + time_overlap_CPU_GPU;
								
							
				}
						
						
			}
					
			//Solapamiento HTD version 2
			if(current_DTH->t_ini == time_counter)
			{
						
						
						if(current_HTD->t_fin < current_DTH->t_fin)	//Solapamos solo una parte de la transferencia HTD.
						{
							
								//time_overlap_GPU_CPU = ((current_HTD->t_fin - time_counter)*overlapped_time_GPU_CPU)/time_GPU_CPU;
								time_overlap_GPU_CPU = ((current_HTD->t_fin - time_counter)*t_overlap_GPU_CPU)/t_GPU_CPU;
							
								time_overlap_GPU_CPU = time_overlap_GPU_CPU + (current_DTH->t_fin - current_HTD->t_fin);
	
						}
						else
						{
							//time_overlap_GPU_CPU = overlapped_time_GPU_CPU;
							time_overlap_GPU_CPU = t_overlap_GPU_CPU;
						
						}
			}
			else
			{
						
						
						if(current_DTH->t_fin < current_HTD->t_fin) //Solapamos solo una parte de la transferencia HTD.
						{
								
								//float completedPercentage = (time_counter - current_DTH->t_ini)/current_DTH->t_estimated_dur;
								
								//float percentageToCompletion = 1 - completedPercentage;
								
								//float timeToCompletion = t_overlap_GPU_CPU*percentageToCompletion;
								
								//time_overlap_GPU_CPU = (time_counter - current_DTH->t_ini) + timeToCompletion;
								
								time_overlap_GPU_CPU = ((current_DTH->t_fin - time_counter)*t_overlap_GPU_CPU)/t_GPU_CPU;
								time_overlap_GPU_CPU = (time_counter - current_DTH->t_ini) + time_overlap_GPU_CPU;
								
							
							
						}
						else
						{
								
								//time_overlap_GPU_CPU = ((current_HTD->t_fin - time_counter)*overlapped_time_GPU_CPU[permutation[current_DTH->id]])/time_GPU_CPU[permutation[current_DTH->id_stream]];
								
								
								time_overlap_GPU_CPU = ((current_HTD->t_fin - time_counter)*t_overlap_GPU_CPU)/t_GPU_CPU;
								time_overlap_GPU_CPU  = (time_counter - current_DTH->t_ini) + time_overlap_GPU_CPU + (current_DTH->t_fin - current_HTD->t_fin);
								
						}
						
						
			}
			
			
			
			
			//Si no es la HTD inicial cuya duracion es 0
			if(current_HTD->t_fin != 0)
				current_HTD->t_fin = current_HTD->t_ini + time_overlap_CPU_GPU;
			
					
			current_HTD->overlapped = true;
					
			//Si no es la DTH inicial cuya duracion es 0
			if(current_DTH->t_fin != 0)
				current_DTH->t_fin = current_DTH->t_ini + time_overlap_GPU_CPU;
					
			
			
			//Si ya se ha planificado una tarea el tiempo de fin unfair solo recogera
			//los tiempos de solapamiento producidos con comandos lanzados anteriormente
			//current_HTD->t_fin_unfair = current_HTD->t_fin;
			//current_DTH->t_fin_unfair = current_DTH->t_fin;
			
			
			
			/*if(current_HTD->launched == true && current_DTH->launched == false && current_DTH->id  == 0)
				current_HTD->t_fin_unfair = current_HTD->t_ini + time_overlap_CPU_GPU;
			
			if(current_DTH->launched == true && current_HTD->launched == false && current_HTD->id == 0)
					current_DTH->t_fin_unfair = current_DTH->t_ini + time_overlap_GPU_CPU;
					
			if(current_HTD->launched == false && current_DTH->launched == true && current_HTD->id  == 0)
					current_HTD->t_fin_unfair = current_HTD->t_ini + time_overlap_CPU_GPU;
			
			if(current_DTH->launched == false && current_HTD->launched == true && current_DTH->id  == 0)
					current_DTH->t_fin_unfair = current_DTH->t_ini + time_overlap_GPU_CPU;*/
			
			current_HTD->t_fin_unfair = current_HTD->t_ini + time_overlap_CPU_GPU;
			current_DTH->t_fin_unfair = current_DTH->t_ini + time_overlap_GPU_CPU;
			
			
			
			current_DTH->overlapped = true;
			
			
#if PRINT_SIMULATOR_TRACE				
				printf("\tHTD - Epoca: %d - Stream: %d - t_ini: %f - t_fin: %f - Duration: %f - Time Overlap CPU GPU: %f\n", current_HTD->id_epoch, current_HTD->id_stream, current_HTD->t_ini, current_HTD->t_fin, current_HTD->t_fin - current_HTD->t_ini, time_overlap_CPU_GPU);
				printf("\tDTH - Epoca: %d - Stream: %d - t_ini: %f - t_fin: %f - Duration: %f - Time Overlap GPU CPU: %f\n", current_HTD->id_epoch, current_DTH->id_stream, current_DTH->t_ini, current_DTH->t_fin, current_DTH->t_fin - current_DTH->t_ini, time_overlap_GPU_CPU);
#endif	
			
					
		}
		
		memset(time_queues, INF_TIME, 3*sizeof(float));
		
		//Cojemos como time_counter el tiempo final de las colas mas pequeño
		if(deque_simulation_HTD.empty())
			time_queues[0] = INF_TIME;
		else
			time_queues[0] = current_HTD->t_fin;
			
		if(deque_simulation_K.empty())
			time_queues[1] = INF_TIME;
		else
			time_queues[1] = current_K->t_fin;
			
		if(deque_simulation_DTH.empty())
			time_queues[2] = INF_TIME;
		else
			time_queues[2] = current_DTH->t_fin;
			
		time_counter = getMinTime(time_queues, 3);

#if PRINT_SIMULATOR_TRACE			
		printf("\nTime Queues:\n");
		printf("HTD: %f\n", time_queues[0]);
		printf("K: %f\n", time_queues[1]);
		printf("DTH: %f\n", time_queues[2]);
		printf("Time Counter; %f\n", time_counter);
#endif

		if(!deque_simulation_HTD.empty() && current_HTD->t_fin == time_counter)
		{	
			//Si no es un HTD perteneciente a la epoca anterior, resolvemos sus dependencias habilitamos el kernel
			//if(current_HTD->id_stream != -1)
				current_HTD->next_command->ready = true;
			
#if PRINT_SIMULATOR_TRACE		
			cout << "\nDesencolando HTD - Epoca " << current_HTD->id_epoch << " - stream " << current_HTD->id_stream << endl;
			cout <<"\tTiempo ini: " << current_HTD->t_ini << " Tiempo fin: " << current_HTD->t_fin << endl;
#endif
			
			if(!deque_simulation_DTH.empty())
				current_DTH->overlapped = false;
				
			
			//if(current_HTD->id == 0 && current_HTD->launched == false)
			if(current_HTD->id == nstreams - 1 && current_HTD->launched == false)
			{
				*t_current_ini_htd = current_HTD->t_ini;
				*t_current_fin_htd = current_HTD->t_fin;
				
				
				//Desencolamos el HTD del stream 0 (tarea primera)
				//encolamos los comandos K y DTH con id_stream == -1
				//que serian los comandos de la epoca anterior que todavia
				//no se han ejecutado
				infoCommand HTD_command;
				
				//HTD
				HTD_command.id_stream = current_HTD->id_stream;
				HTD_command.id = -1;
				HTD_command.id_epoch = current_HTD->id_epoch;
				HTD_command.ready = current_HTD->ready;
				HTD_command.overlapped = current_HTD->overlapped;
				HTD_command.t_ini = current_HTD->t_ini;
				//Cuando queremos meter un comando en la cola current (posterior cola ejecucion)
				//su tiempo de fin tiene que ser el tiempo de fin unfair, ya que este tiempo 
				//solo refleja los incrementos ocasionados con comando lanzados anteriormente
				//HTD_command.t_fin = current_HTD->t_fin;
				HTD_command.t_fin = current_HTD->t_fin_unfair;
				HTD_command.t_fin_unfair = current_HTD->t_fin_unfair;
				HTD_command.enqueue = true;
				HTD_command.launched = true;
				HTD_command.t_CPU_GPU = current_HTD->t_CPU_GPU;
				HTD_command.t_overlap_CPU_GPU = current_HTD->t_overlap_CPU_GPU;
				

#if PRINT_SIMULATOR_TRACE		
			cout << "\nEncolando HTD - Epoca: " <<  HTD_command.id_epoch << " - stream " << HTD_command.id_stream << " en la cola CURRENT HTD" << endl;
#endif
				deque_current_HTD.push_back(HTD_command);
				
				for(deque<infoCommand>::iterator it_K = deque_simulation_K.begin(); 
					it_K != deque_simulation_K.end();
					it_K++)
				{
					//Si quedan comandos de kernels por desencolar, activamos su flag de encolamiento
					//en las colas de CURRENT para tenerlos en cuenta en las proximas epocas
					
						
						it_K->enqueue = true;
					
						
						
					
					
					
				}
				
				//Activamos el flag de encolamiento para el kernel perteneciente a la actual HTD
				current_HTD->next_command->enqueue = true;
				
				//DTH
				for(deque<infoCommand>::iterator it_DTH = deque_simulation_DTH.begin(); 
					it_DTH != deque_simulation_DTH.end();
					it_DTH++)
				{
					//Si quedan comandos de DTH por desencolar, activamos su flag de encolamiento
					//en las colas de CURRENT para tenerlos en cuenta en las proximas epocas
						
						
						it_DTH->enqueue = true;
					
					
				}
				
				//Activamos el flag de encolamiento para la DTH
				current_HTD->next_command->next_command->enqueue = true;
				
				//
			}
				
			//Si es una HTD de la epoca anterior, es posible que se haya incrementado al solaparse
			//con una DTH de la epoca actual, si es asi modificamos los tiempos de inicio y fin
			// de las HTD de la epoca anterior

			//Si es una HTD de la epoca anterior
			if(current_HTD->id == -1)
			{
				//Modificamos los tiempos de inicio de las HTD pertenecientes
				//a la epoca anterior
				float t_fin_htd = current_HTD->t_fin;

				for(deque<infoCommand>::iterator it_HTD = current_HTD + 1; 
					it_HTD != deque_simulation_HTD.end();
					it_HTD++)
				{
					if(it_HTD->id == -1 && it_HTD->t_ini < t_fin_htd)
					{
						float t_dur = it_HTD->t_fin - it_HTD->t_ini;

						it_HTD->t_ini = t_fin_htd;
						it_HTD->t_fin = it_HTD->t_ini + t_dur;

						
					}

					t_fin_htd = it_HTD->t_fin;
					
				}
			}
			//Desencolamos
			deque_simulation_HTD.pop_front();
			
		}
		
		if(!deque_simulation_K.empty() && current_K->t_fin == time_counter)
		{	
			//Resolvemos sus dependencias si no es un kernel de la epoca anterior
			//if(current_K->id_stream != -1)
				current_K->next_command->ready = true;

#if PRINT_SIMULATOR_TRACE	
			cout << "\nDesencolando K - Epoca: " << current_K->id_epoch << " - Stream " << current_K->id_stream << endl;
			cout <<"\tTiempo ini: " << current_K->t_ini << " Tiempo fin: " << current_K->t_fin << endl;
#endif			
			
			
			//if(current_K->id == 0 && current_K->launched == false)
			if(current_K->id == nstreams - 1 && current_K->launched == false)
			{
				*t_current_ini_kernel = current_K->t_ini;
				*t_current_fin_kernel = current_K->t_fin;
			}
			
			//cout << "\tCurrent Ini K: " << *t_current_ini_kernel << " - Current Fin K: " << *t_current_fin_kernel << endl;
				
				
			//Comprobamos si el flag de encolamiento en las colas CURRENT esta activo
			if(current_K->enqueue == true)
			{
				infoCommand kernel_command;
				
				kernel_command.id_stream = current_K->id_stream;
				kernel_command.id = -1;
				kernel_command.id_epoch = current_K->id_epoch;
				kernel_command.ready = current_K->ready;
				kernel_command.t_ini = current_K->t_ini;
				kernel_command.t_fin = current_K->t_fin;
				kernel_command.t_kernel = current_K->t_kernel;
				kernel_command.launched = true;
				
#if PRINT_SIMULATOR_TRACE		
			cout << "\nEncolando K - Epoca: " << kernel_command.id_epoch << " . stream " << kernel_command.id_stream << " en la cola CURRENT K" << endl;
#endif

				deque_current_K.push_back(kernel_command);
			}
			
			//Desencolamos
			deque_simulation_K.pop_front();
		}
		
		if(!deque_simulation_DTH.empty() && current_DTH->t_fin == time_counter)
		{	
			
			//Desencolamos
#if PRINT_SIMULATOR_TRACE
			cout << "\nDesencolando DTH - Epoca: " << current_DTH->id_epoch << " - stream " << current_DTH->id_stream << " Proceso " << current_DTH->id_task << endl;
			cout <<"\tTiempo ini: " << current_DTH->t_ini << " Tiempo fin: " << current_DTH->t_fin << endl;
#endif			
			//Reseteamos el flag de solapamiento de la actual HTD
			if(!deque_simulation_HTD.empty())
				current_HTD->overlapped = false;
				
			
			if(current_DTH->launched == false)
				t_current_last_dth_stream[current_DTH->id_task] = current_DTH->t_fin_unfair;
			
			//if(current_DTH->id == 0 && current_DTH->launched == false)
			if(current_DTH->id == nstreams - 1 && current_DTH->launched == false)
			{
				*t_current_ini_dth = current_DTH->t_ini;
				*t_current_fin_dth = current_DTH->t_fin_unfair; 
				
			}
			
			//Comprobamos si el flag de encolamiento en las colas CURRENT esta activo
			if(current_DTH->enqueue == true)
			{
				infoCommand DTH_command;
				
				DTH_command.id_stream = current_DTH->id_stream;
				DTH_command.id = -1;
				DTH_command.id_epoch = current_DTH->id_epoch;
				DTH_command.id_task = current_DTH->id_task;
				DTH_command.ready = current_DTH->ready;
				DTH_command.enqueue = true;
				DTH_command.launched = true;
				DTH_command.overlapped = false; //current_DTH->overlapped;
				DTH_command.t_ini = current_DTH->t_ini;
				//Cuando queremos meter un comando en la cola current (posterior cola ejecucion)
				//su tiempo de fin tiene que ser el tiempo de fin unfair, ya que este tiempo 
				//solo refleja los incrementos ocasionados con comando lanzados anteriormente
				//DTH_command.t_fin = current_DTH->t_fin;
				DTH_command.t_fin = current_DTH->t_fin_unfair;
				DTH_command.t_fin_unfair = current_DTH->t_fin_unfair;
				DTH_command.t_GPU_CPU = current_DTH->t_GPU_CPU;
				DTH_command.t_overlap_GPU_CPU = current_DTH->t_overlap_GPU_CPU;
				
#if PRINT_SIMULATOR_TRACE		
			cout << "\nEncolando DTH  - Epoca: " << DTH_command.id_epoch << " - stream " << DTH_command.id_stream << " en la cola CURRENT DTH" << endl;
#endif

				deque_current_DTH.push_back(DTH_command);
			}
			
			//Comprobamos si el DTH tiene activo su flag de dependencia. Si tiene activo este flag
			//significa que tiene que activar a un HTD
			if(current_DTH->active_htd == true)
			{
				
				current_DTH->next_command->ready = true;
				current_DTH->active_htd = false;
			}
				
			//Si es una DTH de la epoca anterior, es posible que se haya incrementado al solaparse
			//con una HTD de la epoca actual, si es asi modificamos los tiempos de inicio y fin
			// de las DTH de la epoca anterior

			//Si es una DTH de la epoca anterior
			if(current_DTH->id == -1)
			{
				//Modificamos los tiempos de inicio de las DTH pertenecientes
				//a la epoca anterior
				float t_fin_dth = current_DTH->t_fin;

				for(deque<infoCommand>::iterator it_DTH = current_DTH + 1; 
					it_DTH != deque_simulation_DTH.end();
					it_DTH++)
				{
					if(it_DTH->id == -1 && it_DTH->t_ini < t_fin_dth)
					{
						float t_dur = it_DTH->t_fin - it_DTH->t_ini;

						it_DTH->t_ini = t_fin_dth;
						it_DTH->t_fin = it_DTH->t_ini + t_dur;

						
					}

					t_fin_dth = it_DTH->t_fin;
					
				}
			}
			
			
				
			deque_simulation_DTH.pop_front();
		}
		
#if PRINT_SIMULATOR_TRACE
		printf("*******************************************\n");
		printf("Pulse una tecla para continuar...\n");
		char c = getchar();
#endif
		i++;
		
		
		
	}
	
	
	
	return time_counter;
	
	
	
	
}

void bubble_sort(float *list, int *index)
{
	float aux;
	int indexAux;

	for (int i = 0 ; i < ( N_TASKS - 1 ); i++){
		for (int j = 0 ; j < N_TASKS - i - 1; j++){
			if (list[j] < list[j+1]){
				aux       = list[j];
				list[j]   = list[j+1];
				list[j+1] = aux;
				
				indexAux       = index[j];
				index[j]   = index[j+1];
				index[j+1] = indexAux;
			}
		}
	}
}

float neh_F3(int *h_order_processes, float *h_time_kernels_tasks_execute, float *estimated_time_HTD_per_stream_execute, 
			float *estimated_time_DTH_per_stream_execute, float *estimated_overlapped_time_HTD_per_stream_execute, 
			float *estimated_overlapped_time_DTH_per_stream_execute, float t_previous_ini_htd, float t_previous_ini_kernel,
			float t_previous_ini_dth, float t_previous_fin_htd, float t_previous_fin_kernel, float t_previous_fin_dth,
			float t_previous_overlap_htd, float t_previous_overlap_dth, float *t_current_ini_htd, float *t_current_ini_kernel, 
			float *t_current_ini_dth, float *t_current_fin_htd, float *t_current_fin_kernel, float *t_current_fin_dth,
			float *t_current_overlap_htd, float *t_current_overlap_dth, float *t_previous_last_dth_stream, float *t_current_last_dth_stream,
			int id_epoch)
{
	/***************** NEH *****************/
	float *total_time_tasks = new float[N_TASKS];
	int *index_tasks = new int[N_TASKS];
	int *order_tasks = new int[N_TASKS];
	float *permutation_times = new float[N_TASKS];
	int iBestTime;
	float bestTime;
	float time_counter = 0;

#if PRINT_NEH_TRACE
	cout << "TIEMPOS DE TAREAS" << endl;
	cout << "-----------------" << endl;
	for(int app = 0; app < N_TASKS; app++){
		cout << "TAREA " << app << endl;
		cout << "-------" << endl;
		cout << "HTD: " << estimated_time_HTD_per_stream_execute[app] << endl;
		cout << "KERNEL: " << h_time_kernels_tasks_execute[app] << endl;
		cout << "DTH: " << estimated_time_DTH_per_stream_execute[app] << endl;
	}
	
	cout << "ORIGINAL" << endl;
	cout << "--------" << endl;
#endif

	for(int app = 0; app < N_TASKS; app++){
		total_time_tasks[app] = estimated_time_HTD_per_stream_execute[app] +
								h_time_kernels_tasks_execute[app] +
								estimated_time_DTH_per_stream_execute[app];
		index_tasks[app] = app;
		
#if PRINT_NEH_TRACE
		cout << index_tasks[app] << ": " << total_time_tasks[app] << endl;
#endif
	}
	
	bubble_sort(total_time_tasks, index_tasks);

#if PRINT_NEH_TRACE	
	cout << "ORDENADO" << endl;
	cout << "--------" << endl;
	for(int app = 0; app < N_TASKS; app++){
		cout << index_tasks[app] << ": " << total_time_tasks[app] << endl;
	}
#endif

	int *new_order_tasks = new int[2];
	
	new_order_tasks[0] = index_tasks[0];
	new_order_tasks[1] = index_tasks[1];
	
	// order_tasks[0] = index_tasks[0];
	// order_tasks[1] = index_tasks[1];

#if PRINT_NEH_TRACE		
	cout << "ORDEN S1" << endl;
	cout << "--------" << endl;
	cout << new_order_tasks[0] << endl;
	cout << new_order_tasks[1] << endl;
#endif
	
	permutation_times[0] = simulator2CopyEngine_bubble_command_v6_FAIR(h_time_kernels_tasks_execute, 1, estimated_time_HTD_per_stream_execute,
								estimated_time_DTH_per_stream_execute, estimated_overlapped_time_HTD_per_stream_execute,
								estimated_overlapped_time_DTH_per_stream_execute, new_order_tasks,
								new_order_tasks, 2,
								t_previous_ini_htd, t_previous_ini_kernel, t_previous_ini_dth,
								t_previous_fin_htd, t_previous_fin_kernel, t_previous_fin_dth,
								t_previous_overlap_htd, t_previous_overlap_dth,
								t_current_ini_htd, t_current_ini_kernel, t_current_ini_dth,
								t_current_fin_htd, t_current_fin_kernel, t_current_fin_dth,
								t_current_overlap_htd, t_current_overlap_dth,
								t_previous_last_dth_stream, t_current_last_dth_stream, id_epoch);

#if PRINT_NEH_TRACE									
	cout << "PERMUTATION TIME" << endl;
	cout << "----------------" << endl;
	cout << permutation_times[0] << endl;
#endif

	new_order_tasks[0] = index_tasks[1];
	new_order_tasks[1] = index_tasks[0];
	
	// order_tasks[0] = index_tasks[1];
	// order_tasks[1] = index_tasks[0];

#if PRINT_NEH_TRACE		
	cout << "ORDEN S2" << endl;
	cout << "--------" << endl;
	cout << new_order_tasks[0] << endl;
	cout << new_order_tasks[1] << endl;
#endif

	permutation_times[1] = simulator2CopyEngine_bubble_command_v6_FAIR(h_time_kernels_tasks_execute, 1, estimated_time_HTD_per_stream_execute,
								estimated_time_DTH_per_stream_execute, estimated_overlapped_time_HTD_per_stream_execute,
								estimated_overlapped_time_DTH_per_stream_execute, new_order_tasks,
								new_order_tasks, 2,
								t_previous_ini_htd, t_previous_ini_kernel, t_previous_ini_dth,
								t_previous_fin_htd, t_previous_fin_kernel, t_previous_fin_dth,
								t_previous_overlap_htd, t_previous_overlap_dth,
								t_current_ini_htd, t_current_ini_kernel, t_current_ini_dth,
								t_current_fin_htd, t_current_fin_kernel, t_current_fin_dth,
								t_current_overlap_htd, t_current_overlap_dth,
								t_previous_last_dth_stream, t_current_last_dth_stream, id_epoch);

#if PRINT_NEH_TRACE									
	cout << "PERMUTATION TIME" << endl;
	cout << "----------------" << endl;
	cout << permutation_times[1] << endl;
#endif
	
	delete [] new_order_tasks;
	
	if(permutation_times[0] < permutation_times[1]){
		order_tasks[0] = index_tasks[0];
		order_tasks[1] = index_tasks[1];
	}
	else{
		order_tasks[0] = index_tasks[1];
		order_tasks[1] = index_tasks[0];
	}
	
#if PRINT_NEH_TRACE	
	cout << "ORDEN S1 O S2" << endl;
	cout << "-------------" << endl;
	cout << order_tasks[0] << endl;
	cout << order_tasks[1] << endl;
#endif
	
	for(int i = 2; i < N_TASKS; i++){
#if PRINT_NEH_TRACE	
		cout << "TASK " << i << endl;
		cout << "******" << endl;
#endif

		int *new_order_tasks = new int[i+1];
		
		for(int k = 0; k <= i; k++){
			for(int j = i; j > k; j--)
				new_order_tasks[j] = order_tasks[j - 1];
				
			new_order_tasks[k] = index_tasks[i];
			
			for(int j = k - 1; j >= 0; j--)
				new_order_tasks[j] = order_tasks[j];

#if PRINT_NEH_TRACE	
			cout << "PERMUTATION" << endl;
			cout << "-----------" << endl;
			for(int j = 0; j <= i; j++)
				cout << new_order_tasks[j] << endl;
#endif	
	
			permutation_times[k] = simulator2CopyEngine_bubble_command_v6_FAIR(h_time_kernels_tasks_execute, 1, estimated_time_HTD_per_stream_execute, 
								estimated_time_DTH_per_stream_execute, estimated_overlapped_time_HTD_per_stream_execute, 
								estimated_overlapped_time_DTH_per_stream_execute, new_order_tasks, 
								new_order_tasks, i+1, 
								t_previous_ini_htd, t_previous_ini_kernel, t_previous_ini_dth,
								t_previous_fin_htd, t_previous_fin_kernel, t_previous_fin_dth,
								t_previous_overlap_htd, t_previous_overlap_dth,
								t_current_ini_htd, t_current_ini_kernel, t_current_ini_dth,
								t_current_fin_htd, t_current_fin_kernel, t_current_fin_dth,
								t_current_overlap_htd, t_current_overlap_dth,
								t_previous_last_dth_stream, t_current_last_dth_stream, id_epoch);

#if PRINT_NEH_TRACE									
			cout << "TIEMPO " << k << endl;
			cout << "--------" << endl;
			cout << permutation_times[k] << endl;
#endif
		}
		
		delete [] new_order_tasks;
		
		iBestTime = 0;
		bestTime = permutation_times[0];
		
		for(int j = 1; j <= i; j++){
			if(bestTime >= permutation_times[j]){
				iBestTime = j;
				bestTime = permutation_times[j];
			}
		}
		
		for(int j = i; j > iBestTime; j--)
			order_tasks[j] = order_tasks[j - 1];
		
		order_tasks[iBestTime] = index_tasks[i];

#if PRINT_NEH_TRACE	
		cout << "MEJOR ORDEN" << endl;
		cout << "-----------" << endl;
		for(int j = 0; j <= i; j++)
			cout << order_tasks[j] << endl;
#endif
	}
	
	/**** AÑADIR AQUI UNA NUEVA SIMULACION PARA TENER LOS ÚLTIMOS DATOS ****/
	permutation_times[0] = simulator2CopyEngine_bubble_command_v6_FAIR(h_time_kernels_tasks_execute, 1, estimated_time_HTD_per_stream_execute, 
								estimated_time_DTH_per_stream_execute, estimated_overlapped_time_HTD_per_stream_execute, 
								estimated_overlapped_time_DTH_per_stream_execute, order_tasks, 
								order_tasks, N_TASKS,
								t_previous_ini_htd, t_previous_ini_kernel, t_previous_ini_dth,
								t_previous_fin_htd, t_previous_fin_kernel, t_previous_fin_dth,
								t_previous_overlap_htd, t_previous_overlap_dth,
								t_current_ini_htd, t_current_ini_kernel, t_current_ini_dth,
								t_current_fin_htd, t_current_fin_kernel, t_current_fin_dth,
								t_current_overlap_htd, t_current_overlap_dth,
								t_previous_last_dth_stream, t_current_last_dth_stream, id_epoch);
								
	time_counter = permutation_times[0];

#if PRINT_NEH_TRACE	
	cout << "ORDEN FINAL" << endl;
	cout << "-----------" << endl;
	for(int i = 0; i < N_TASKS; i++)
		cout << order_tasks[i] << endl;
#endif
		
	for(int i = 0; i < N_TASKS; i++)
		h_order_processes[i] = order_tasks[i];

#if PRINT_NEH_TRACE	
	cout << "ORDEN TASKS" << endl;
	cout << "-----------" << endl;
	for(int i = 0; i < N_TASKS; i++)
		cout << h_order_processes[i] << endl;
#endif
	
	delete [] total_time_tasks;
	delete [] index_tasks;
	delete [] order_tasks;
	delete [] permutation_times;
	/***************************************/
	
	return time_counter;
}

float heuristic(int *h_order_processes, int n_app, int *execute_batch, float *time_kernel_tasks, float *time_HTD_tasks, 
				float *time_DTH_tasks, float *time_overlapped_HTD_tasks, float *time_overlapped_DTH_tasks,
				int scheduling_batch, 
				float t_previous_ini_htd, float t_previous_ini_kernel, float t_previous_ini_dth,
				float t_previous_fin_htd, float t_previous_fin_kernel, float t_previous_fin_dth,
				float *t_current_ini_htd, float *t_current_ini_kernel, float *t_current_ini_dth,
				float *t_current_fin_htd, float *t_current_fin_kernel, float *t_current_fin_dth,
				float *t_previous_last_dth_stream, float *t_current_last_dth_stream, int id_epoch)
{
	int nstreams = n_app;	//Number of tasks in batch = nstreams
	
	infoCommand kernel_command;		//command structure (Kernel)
	infoCommand HTD_command;		//command structure (HTD)
	infoCommand DTH_command;		//command structure (DTH)
	float time_counter = 0;			
	
	//n_type_tasks = total_tasks
	int total_tasks = scheduling_batch;	//Number of tasks in the scheduling batch
	
	float time_counter_HTD = 0.0;		//Time counter of the HTD commands
	float time_counter_K   = 0.0;		//Time counter of the Kernel commands
	float time_counter_DTH = 0.0;		//Time counter of th DTH commands
	float time_queues[3];				//Finish time of the queues.
	                     				
	///array of selected Task. This array stores the index of the selected tasks.
	int *h_tasks_pattern      = new int[total_tasks];
	//Array of the type of tasks.	
	int *h_type_dominant      = new int[total_tasks];
	//Array of HTD times for the heuristic. The heuristic has to consider the delay time of each stream.
	//This delay time is added to the current HTD time of the launched task in the stream.
	float *htd_time_heuristic = new float[total_tasks];

	
	//Array of inserted tasks. An element in the index i of this array is to set 1 if the task with index i has been inserted (selected).
	int  *inserted_tasks_type = new int[total_tasks];
	memset(inserted_tasks_type , 0, total_tasks * sizeof(int));
	
	//Classifing the task in DK (dominant kernel) or DT (dominant transfer)
	for(int t = 0; t < total_tasks; t++)
	{
		if(time_kernel_tasks[t] >= time_HTD_tasks[t] + time_DTH_tasks[t])
			h_type_dominant[t] = DK;
		else
			h_type_dominant[t] = DT;
	}


	//Adding the delay time of the stream to the HTD time of the task.
	for(int i = 0; i < total_tasks; i++)
	{
		htd_time_heuristic[i] = time_HTD_tasks[i] + t_current_last_dth_stream[execute_batch[i]];
	}


	//Searching the tasks with the longest DTH time.
	int task_max_dth = getMaxTask(time_DTH_tasks, inserted_tasks_type, total_tasks);
	int task_min_kdth = -1;
	
	float *time_KDTH_tasks = new float[total_tasks];
	for(int i= 0; i < total_tasks; i++)
		time_KDTH_tasks[i] = time_kernel_tasks[i] + time_DTH_tasks[i];

	//Obtenemos el tiempo de K+DTH menor
	float min_time_kdth = getMinTime(time_KDTH_tasks, total_tasks);
	//Contamos cuantas tareas tienen k+dth igual a min_time_dth
	vector<int>v_tasks_min_kdth;
	for(int t = 0; t < total_tasks; t++)
		if(time_KDTH_tasks[t] == min_time_kdth)
			v_tasks_min_kdth.push_back(t);

	delete [] time_KDTH_tasks;

	if(v_tasks_min_kdth.size() > 1)
	{
		//Si hay mas de una tarea con su tiempo de k+dth igual a min_time_dth
		//Seleccionamos aquella que tenga menor HTD
		int htd_min_time = 100000;
		for(int i = 0; i < v_tasks_min_kdth.size(); i++)
			if(htd_min_time > htd_time_heuristic[v_tasks_min_kdth.at(i)])
			{
				htd_min_time = htd_time_heuristic[v_tasks_min_kdth.at(i)];
				task_min_kdth = v_tasks_min_kdth.at(i);
			}
	}
	else
		task_min_kdth = v_tasks_min_kdth.at(0);


	//Loop the all tasks.
	for(int t_tasks = 0; t_tasks < total_tasks; t_tasks++)
	{
		//Cleanning the simulation queues.
		deque_simulation_HTD.clear();
		deque_simulation_K.clear();
		deque_simulation_DTH.clear();
		//Cleanning  the current queues.
		deque_current_HTD.clear();
		deque_current_K.clear();
		deque_current_DTH.clear();
	
		//Inserting into the simulation queues the commands from the execution queues. Theses commands are belonging to
		//a previous epoch or task group.
		//HTD
		for(deque<infoCommand>::iterator current_HTD = deque_execution_HTD.begin(); current_HTD != deque_execution_HTD.end(); current_HTD++)
		{
			
			
			HTD_command.id_stream         = current_HTD->id_stream;
			HTD_command.id_epoch          = current_HTD->id_epoch;
			HTD_command.id                = current_HTD->id;	
			HTD_command.ready             = true;
			HTD_command.overlapped        = false;
			HTD_command.enqueue           = false;
			HTD_command.launched          = true;
			HTD_command.t_ini             = current_HTD->t_ini;
			HTD_command.t_fin             = current_HTD->t_fin;
			HTD_command.t_estimated_fin   = current_HTD->t_estimated_fin;
			HTD_command.t_CPU_GPU         = current_HTD->t_CPU_GPU;
			HTD_command.t_overlap_CPU_GPU = current_HTD->t_overlap_CPU_GPU;
			
			deque_simulation_HTD.push_back(HTD_command);
			
		}
		
		//KERNEL
		for(deque<infoCommand>::iterator current_K = deque_execution_K.begin(); current_K != deque_execution_K.end(); current_K++)
		{
			
			kernel_command.id_stream = current_K->id_stream;
			kernel_command.id        = current_K->id;
			kernel_command.id_epoch  = current_K->id_epoch;
			kernel_command.ready     = false;
			kernel_command.enqueue   = false;
			kernel_command.launched  = true;
			kernel_command.t_ini     = current_K->t_ini;
			kernel_command.t_fin     = current_K->t_fin;
			kernel_command.t_kernel  = current_K->t_kernel;
			
			deque_simulation_K.push_back(kernel_command);
			
		}
	
		for(deque<infoCommand>::iterator current_DTH = deque_execution_DTH.begin(); current_DTH != deque_execution_DTH.end(); current_DTH++)
		{
			
			DTH_command.id_stream         = current_DTH->id_stream;
			DTH_command.id                = current_DTH->id;
			DTH_command.id_epoch          = current_DTH->id_epoch;
			DTH_command.ready             = false;
			DTH_command.overlapped        = false;
			DTH_command.enqueue           = false;
			DTH_command.launched          = true;
			DTH_command.active_htd        = false;
			DTH_command.t_ini             = current_DTH->t_ini;
			DTH_command.t_fin             = current_DTH->t_fin;
			DTH_command.t_estimated_fin   = current_DTH->t_estimated_fin;
			DTH_command.t_GPU_CPU         = current_DTH->t_GPU_CPU;
			DTH_command.t_overlap_GPU_CPU = current_DTH->t_overlap_GPU_CPU;
			
			deque_simulation_DTH.push_back(DTH_command);
			
		}
		
	
	
		int num_HTD_enqueue = deque_simulation_HTD.size();
		int num_DTH_enqueue = deque_simulation_DTH.size();
		int num_K_enqueue   = deque_simulation_K.size();
		
		
		
		//Setting the dependences among the commands of the previous epoch.
		//HTD->K
		for(deque<infoCommand>::iterator current_K = deque_simulation_K.begin(); 
			current_K != deque_simulation_K.end(); current_K++)
		{
			
			current_K->ready = true;
			for(deque<infoCommand>::iterator current_HTD = deque_simulation_HTD.begin(); 
				current_HTD != deque_simulation_HTD.end(); current_HTD++)
			{
				if(current_HTD->id_stream == current_K->id_stream && current_HTD->id_epoch == current_K->id_epoch)
				{				
					current_K->ready = false;
					current_HTD->next_command = current_K;
					
				}
				
				
			}
			
			
			
		}
		
		//K->DTH
		for(deque<infoCommand>::iterator current_DTH = deque_simulation_DTH.begin(); 
			current_DTH != deque_simulation_DTH.end(); current_DTH++)
		{
			
			current_DTH->ready = true;
			for(deque<infoCommand>::iterator current_K = deque_simulation_K.begin(); 
				current_K != deque_simulation_K.end(); current_K++)
			{
				if(current_K->id_stream == current_DTH->id_stream && current_K->id_epoch == current_DTH->id_epoch)
				{
					current_DTH->ready = false;
					current_K->next_command = current_DTH;
					
				}
				
				
			}
			
			
			
		}


		int idx_selected_task = -1;
		
		//Number of DK Tasks = Number of DT Tasks
		if(getNumberTypeDominant(h_type_dominant, total_tasks, DT) 
		== getNumberTypeDominant(h_type_dominant, total_tasks, DK))
		{
			//cout << "\tDK == DT" << endl;
#if PRUEBA_KDTH_MIN
			int first_task;
#endif
			if(t_tasks == 0)
			{
				
				//Buscamos el tiempo de HTD menor de las tareas DK
					float min_htd_time = 1000000;

					for(int t = 0; t < total_tasks; t++)
						if(htd_time_heuristic[t] < min_htd_time && h_type_dominant[t] == DK)
							min_htd_time = htd_time_heuristic[t];

					//Contamos cuantas tareas tienen el tiempo de htd minimo
					vector<int>v_tasks_min_htd;
					for(int t = 0; t < total_tasks; t++)
						if(htd_time_heuristic[t] == min_htd_time && h_type_dominant[t] == DK)
							v_tasks_min_htd.push_back(t);



					if(v_tasks_min_htd.size() > 1)
					{
						//Si mas de una tarea con el tiempo de htd igual a min_htd_time
						//seleccionamos aquella que tenga mas tiempo de dth

						float max_time_dth = 0;
						for(int t = 0; t < v_tasks_min_htd.size(); t++)
							if(max_time_dth < time_DTH_tasks[v_tasks_min_htd.at(t)])
								max_time_dth = time_DTH_tasks[v_tasks_min_htd.at(t)];

						//Contamos cuantas tareas del vector v_tasks_min_htd tienen
						//el tiempo de DTH igual a max_time_dth
						vector<int>v_tasks_max_dth;

						for(int t = 0; t < v_tasks_min_htd.size(); t++)
							if(max_time_dth == time_DTH_tasks[v_tasks_min_htd.at(t)])
								v_tasks_max_dth.push_back(v_tasks_min_htd.at(t));

						if(v_tasks_max_dth.size() > 1)
						{
							//Si existe mas de una tarea con el mismo tiempo de dth igual a max_time_dth
							//seleccionamos aquella que tenga mayor kernel
							float max_time_kernel = 0;
							for(int t = 0; t < v_tasks_max_dth.size(); t++)
							{
								if(max_time_kernel < time_kernel_tasks[v_tasks_max_dth.at(t)])
								{
									max_time_kernel = time_kernel_tasks[v_tasks_max_dth.at(t)];
									idx_selected_task = v_tasks_max_dth.at(t);
								}
							}
						}
						else
							idx_selected_task = v_tasks_max_dth.at(0);

					}
					else
						idx_selected_task = v_tasks_min_htd.at(0);

				

#if PRUEBA_KDTH_MIN
					first_task = idx_selected_task;
#endif
				/*int num_tasks_DK = getNumberTypeDominant(h_type_dominant, total_tasks, DK);
				float *sorted_tasks_DK = new float [num_tasks_DK];
				float *htd_time_DK = new float[num_tasks_DK];
				
				
				int t = 0;
				
				for(int i = 0; i < total_tasks; i++)
				{
					if(h_type_dominant[i] == DK)
					{
						sorted_tasks_DK[t] = (float)i;
						//htd_time_DK[t] = time_HTD_tasks[i];
						htd_time_DK[t] = htd_time_heuristic[i];
						t++;
						
					}
					
					
				}

				serialQuickSortByKeyG(htd_time_DK, sorted_tasks_DK, num_tasks_DK);

				//Buscamos la tarea DK con menor DTH la cual no nos interesara
				//colocarla en primer lugar, mas bien intentaremos colocarla en ultimo lugar
				//para minimizar el tiempo de transferencia DTH final
				float time = 1000000;
				int taskDK_min_dth = -1;
				

				taskDK_min_dth = getTaskNoFirst(time_DTH_tasks, htd_time_heuristic, time_kernel_tasks, inserted_tasks_type, h_type_dominant, DK, total_tasks);

				t = 0;
				bool found = false;
				int task;
				while(found == false && t < num_tasks_DK)
				{
					task = (int) sorted_tasks_DK[t];
					
					
	
					//Si la tarea no es la que menor DTH. Ya que no interesa
					//intentar colocar la tarea que menor DTH al final para minimizar
					//la transferencia DTH final
					if(taskDK_min_dth != task)
					{
						found = true;
						for(int j = 0; j < total_tasks; j++)
						{
							if(j != task
							&& inserted_tasks_type[j] != 1
							&& htd_time_heuristic[j] > time_kernel_tasks[task])
							//&& time_HTD_tasks[j] > time_kernel_tasks[task])
							{
								found = false;
								
								
							}
						}
					}
					
					t++;
					
					
					
				}

				if(found == true)
				{
					
					idx_selected_task = task;
				}
				else
				{
						
						
						t = 0;
						bool found = false;
						int task;
						while(found == false && t < num_tasks_DK)
						{
							task = (int) sorted_tasks_DK[t];
					
					
	
							//Si la tarea no es la que menor DTH. Ya que no interesa
							//intentar colocar la tarea que menor DTH al final para minimizar
							//la transferencia DTH final
						
							found = true;
							for(int j = 0; j < total_tasks; j++)
							{
								if(j != task
								&& inserted_tasks_type[j] != 1
								&& htd_time_heuristic[j] > time_kernel_tasks[task])
								//&& time_HTD_tasks[j] > time_kernel_tasks[task])
								{
									found = false;
								
								
								}
							}
							t++;
					
					
					
						}
						
						if(found == true)
						{
					
							idx_selected_task = task;
						}
						else
						{
							
							float time = 1000000000;

							for(int t = 0; t < total_tasks; t++)
							{
					
								//if(time_HTD_tasks[t] < time
								if(htd_time_heuristic[t] < time  
								&& inserted_tasks_type[t] != 1
								&& h_type_dominant[t] == DK)
								{
									//time = time_HTD_tasks[t];
									time = htd_time_heuristic[t];
									idx_selected_task = t;
									//inserted_type = DK;
						
						
								}
						
							}
							
							
						}
					
				
				}
				
				
				delete [] sorted_tasks_DK;
				delete [] htd_time_DK;

#if PRUEBA_KDTH_MIN
				first_task = idx_selected_task;
#endif*/
			}
			else
			{

#if PRUEBA_KDTH_MIN	
				if(first_task != task_min_kdth)
				{
					inserted_tasks_type[task_min_kdth] = 1;
					if(t_tasks == total_tasks - 1)
						inserted_tasks_type[task_min_kdth] = 0;
				}
#endif

				int num_tasks = 0;
				bool flag = false;
				for(int t = 0; t < total_tasks; t++)
				{
					
					
					//if((time_counter_HTD + time_HTD_tasks[t] <= time_counter_K)
					if((time_counter_HTD + htd_time_heuristic[t] <= time_counter_K)  
					&& (inserted_tasks_type[t] != 1))
					{
							
							num_tasks++;
					}
						
					
				}

				if(num_tasks > 0)
				{
					//Existe mas de una tarea que time_counter_HTD + HTD, no sobrepasa
					//a time_counter_HTD + time_counter_K
					
					//Guardamos su tipo
					int *type_non_overlap = new int[num_tasks];
					int i = 0;
					for(int t = 0; t < total_tasks; t++)
					{
					
							//if((time_counter_HTD + time_HTD_tasks[t] <= time_counter_K)
							if((time_counter_HTD + htd_time_heuristic[t] <= time_counter_K)  
							&& (inserted_tasks_type[t] != 1))
							{
								
								type_non_overlap[i] = t;
								i++;
							}
						
					
					}
						
					float time = 1000000;
						
					//De las tareas cuya HTD cuyo time_counter_HTD + HTD, no sobrepasa
					//a time_counter_HTD + time_counter_K, vemos aquellas cuya insercion
					//genera un tiempo menor
						
					for(int i = 0; i < num_tasks; i++)
					{
							//Vemos si al insertar la nueva tarea podriamos solapar la DTH anterior.
							if(time_counter_DTH >= time_counter_K + time_kernel_tasks[type_non_overlap[i]]
							&& (inserted_tasks_type[type_non_overlap[i]] != 1))
							{
								//Si no podemos solaparla, el tiempo se incrementaria en time_counter_DTH + time_DTH_tasks[type_non_overlap[i]];
								if(time > time_counter_DTH + time_DTH_tasks[type_non_overlap[i]])
								{
										time = time_counter_DTH + time_DTH_tasks[type_non_overlap[i]];
										idx_selected_task = type_non_overlap[i];
										flag = true;
										
								}
								
							}
							
							//Si podemos solapar la DTH anterior
							if(time_counter_DTH < time_counter_K + time_kernel_tasks[type_non_overlap[i]]
							&& (inserted_tasks_type[type_non_overlap[i]] != 1))
							{
								//El tiempo se veria incrementado en time_counter_K + time_kernel_tasks[type_non_overlap[i]] + time_DTH_tasks[type_non_overlap[i]]
								if(time > time_counter_K + time_kernel_tasks[type_non_overlap[i]] + time_DTH_tasks[type_non_overlap[i]])
								{
									time = time_counter_K + time_kernel_tasks[type_non_overlap[i]] + time_DTH_tasks[type_non_overlap[i]];
									idx_selected_task = type_non_overlap[i];
									flag = true;
									
								}
								
							}
								
							
							
					}
							
					//Si los kernels de todas las tareas solapan al time_counter_DTH cojemos aquella cuyo DTH sea mayor
					if(flag == false)
					{
							int max = -1;
							
							for(i = 0; i < num_tasks; i++)
							{
								
								if(time_DTH_tasks[type_non_overlap[i]] > max)
								{
									idx_selected_task = type_non_overlap[i];
									max = time_DTH_tasks[type_non_overlap[i]];
									
								}
								
							}
							
							
					}
							
							
						
					delete [] type_non_overlap;
					

					
				}
				else
				{
						
						//Vemos aquella tarea cuya insercion produca un tiempo menor
						float time = 10000000;
						int index;
						
						
						
						
						//Si para todas las tareas por insertar la suma time_counter_HTD + HTD
						//sobrepasa la suma time_counter_HTD + time_counter_K, seleccionamos
						//aquella tarea cuya diferencia (time_counter_HTD + HTD) - time_counter_K
						//sea menor
						
						for(int t = 0; t < total_tasks; t++)
						{
							//if((time > time_counter_HTD + time_HTD_tasks[t] - time_counter_K)
							if((time > time_counter_HTD + htd_time_heuristic[t] - time_counter_K)
							&& (inserted_tasks_type[t] != 1))
							{
								time = time_counter_HTD + time_HTD_tasks[t] - time_counter_K;
								idx_selected_task = t;
								
							}
							
							
						}
					
					
					
				}
				
			
				//Si es la penultima tarea vemos si la tarea con mayor
				//DTH se ha insertado. Si no se ha insertado la insertamos.
				//Si ya se ha insertado seguimos con la seleccion.
				
				if(t_tasks == total_tasks - 2)
				//if(t_tasks == (N_SINTETIC_TASKS/2) - 1)
				{
					if(inserted_tasks_type[task_max_dth] != 1)
						idx_selected_task = task_max_dth;
					
				}
			}

		}

		//Si el numero de tareas DT es mayor que el numero de tareas DK
		if(getNumberTypeDominant(h_type_dominant, total_tasks, DT) 
		> getNumberTypeDominant(h_type_dominant, total_tasks, DK))
		{
			//cout << "\tDK < DT" << endl;
#if PRUEBA_KDTH_MIN
			int first_task;
#endif
			if(t_tasks == 0)
			{

					//Buscamos el tiempo de HTD menor
				float min_htd_time = 1000000;
				//Contamos cuantas tareas tienen el tiempo de htd minimo
				vector<int>v_tasks_min_htd;

				if(getNumberTypeDominant(h_type_dominant, total_tasks, DK) != 0)
				{
					
					for(int t = 0; t < total_tasks; t++)
						if(htd_time_heuristic[t] < min_htd_time && h_type_dominant[t] == DK)
							min_htd_time = htd_time_heuristic[t];

					
					for(int t = 0; t < total_tasks; t++)
						if(htd_time_heuristic[t] == min_htd_time && h_type_dominant[t] == DK)
							v_tasks_min_htd.push_back(t);
				}
				else
				{
					min_htd_time = getMinTime(htd_time_heuristic, total_tasks);
					for(int t = 0; t < total_tasks; t++)
						if(htd_time_heuristic[t] == min_htd_time)
							v_tasks_min_htd.push_back(t);
				}



					if(v_tasks_min_htd.size() > 1)
					{
						//Si mas de una tarea con el tiempo de htd igual a min_htd_time
						//seleccionamos aquella que tenga mas tiempo de dth

						float max_time_dth = 0;
						for(int t = 0; t < v_tasks_min_htd.size(); t++)
							if(max_time_dth < time_DTH_tasks[v_tasks_min_htd.at(t)])
								max_time_dth = time_DTH_tasks[v_tasks_min_htd.at(t)];

						//Contamos cuantas tareas del vector v_tasks_min_htd tienen
						//el tiempo de DTH igual a max_time_dth
						vector<int>v_tasks_max_dth;

						for(int t = 0; t < v_tasks_min_htd.size(); t++)
							if(max_time_dth == time_DTH_tasks[v_tasks_min_htd.at(t)])
								v_tasks_max_dth.push_back(v_tasks_min_htd.at(t));

						if(v_tasks_max_dth.size() > 1)
						{
							//Si existe mas de una tarea con el mismo tiempo de dth igual a max_time_dth
							//seleccionamos aquella que tenga mayor kernel
							float max_time_kernel = 0;
							for(int t = 0; t < v_tasks_max_dth.size(); t++)
							{
								if(max_time_kernel < time_kernel_tasks[v_tasks_max_dth.at(t)])
								{
									max_time_kernel = time_kernel_tasks[v_tasks_max_dth.at(t)];
									idx_selected_task = v_tasks_max_dth.at(t);
								}
							}
						}
						else
							idx_selected_task = v_tasks_max_dth.at(0);

					}
					else
						idx_selected_task = v_tasks_min_htd.at(0);

				

#if PRUEBA_KDTH_MIN
					first_task = idx_selected_task;
#endif
				

			}
			else
			{
#if PRUEBA_KDTH_MIN
				
				if(first_task != task_min_kdth)
				{
					inserted_tasks_type[task_min_kdth] = 1;
					if(t_tasks == total_tasks - 1)
						inserted_tasks_type[task_min_kdth] = 0;
				}
					
#endif

				//Si no es la tarea primera
				//Buscamos tareas DK que time_counter_HTD + HTD, no sobrepase
				//a time_counter_HTD + time_counter_K
					
				int num_tasks = 0;
				bool flag = false;
				for(int t = 0; t < total_tasks; t++)
				{
					
					
						//if((time_counter_HTD + time_HTD_tasks[t] <= time_counter_K)
						if((time_counter_HTD + htd_time_heuristic[t] <= time_counter_K)  
						&& (inserted_tasks_type[t] != 1))
						//&& h_type_dominant[t] == DT)
						{
							
							num_tasks++;
						}
						
					
				}

				if(num_tasks > 0)
				{
						//Existe mas de una tarea que time_counter_HTD + HTD, no sobrepasa
						//a time_counter_HTD + time_counter_K
						
					
						//Guardamos su tipo
						int *type_non_overlap = new int [num_tasks];
						int i = 0;
						for(int t = 0; t < total_tasks; t++)
						{
					
							//if((time_counter_HTD + time_HTD_tasks[t] <= time_counter_K)
							if((time_counter_HTD + htd_time_heuristic[t] <= time_counter_K)  
							&& (inserted_tasks_type[t] != 1))
							//&& h_type_dominant[t] == DT)
							{
								
								type_non_overlap[i] = t;
								//cout << "T_task: " << t_tasks << "T: " << t << " HTD: " << time_HTD_tasks[t] << " TK: " << time_counter_K <<  " THTD: " << time_counter_HTD << " Diferencia: " << time_counter_K - (time_counter_HTD + time_HTD_tasks[t]) << endl;
								i++;
							}
						
					
						}
						
						
						float time = 1000000;
						
						//De las tareas cuya HTD cuyo time_counter_HTD + HTD, no sobrepasa
						//a time_counter_HTD + time_counter_K, vemos aquellas cuya insercion
						//genera un tiempo menor
						
						//cout << "T_task: " << t_tasks << endl;
						//cout << "\t" << endl;
						for(int i = 0; i < num_tasks; i++)
						{
							
							//De las tareas DK cuyo time_counter_HTD + time_HTD no sobrepasa al time_counter_K
							//Seleccionamos aquella cuya diferencia time_counter_K - time_counter_HTD + time_HTD
							//sea menor
							
							//cout << "Task: " << type_non_overlap[i] << " HTD: " << time_HTD_tasks[type_non_overlap[i]] << " TK: " << time_counter_K <<  " THTD: " << time_counter_HTD << " Diferencia: " << time_counter_K - (time_counter_HTD + time_HTD_tasks[type_non_overlap[i]]) << endl;
							
							//if(time > (time_counter_K - (time_counter_HTD + time_HTD_tasks[type_non_overlap[i]])))
							if(time > (time_counter_K - (time_counter_HTD + htd_time_heuristic[type_non_overlap[i]])))
							{
								//time = time_counter_K - (time_counter_HTD + time_HTD_tasks[type_non_overlap[i]]);
								time = time_counter_K - (time_counter_HTD + htd_time_heuristic[type_non_overlap[i]]);
								idx_selected_task = type_non_overlap[i];
								flag = true;
								
							}
							
							
							
						}
						
						
						
						
						//Si los kernels de todas las tareas solapan al time_counter_DTH cojemos aquella cuyo DTH sea mayor
						if(flag == false)
						{
							int max = -1;
							
							for(i = 0; i < num_tasks; i++)
							{
								
								if(time_DTH_tasks[type_non_overlap[i]] > max)
								{
									idx_selected_task = type_non_overlap[i];
									max = time_DTH_tasks[type_non_overlap[i]];
									
								}
								
							}
							
							
						}
							
						delete [] type_non_overlap;
					
				}
				else
				{
						
						//Vemos aquella tarea cuya insercion produca un tiempo menor
						float time = 10000000;
						int index;
						
						
						//Si para todas las tareas por insertar la suma time_counter_HTD + HTD
						//sobrepasa la suma time_counter_HTD + time_counter_K, seleccionamos
						//aquella tarea cuya diferencia (time_counter_HTD + HTD) - time_counter_K
						//sea menor
						
						
						for(int t = 0; t < total_tasks; t++)
						{
							//if((time > time_counter_HTD + time_HTD_tasks[t] - time_counter_K)
							if((time > time_counter_HTD + htd_time_heuristic[t] - time_counter_K)
							&& (inserted_tasks_type[t] != 1))
							{
								//time = time_counter_HTD + time_HTD_tasks[t] - time_counter_K;
								time = time_counter_HTD + htd_time_heuristic[t] - time_counter_K;
								idx_selected_task = t;
								
							}
							
							
						}
					
					
					
				}
				
			
				//Si es la penultima tarea vemos si la tarea con mayor
				//DTH se ha insertado. Si no se ha insertado la insertamos.
				//Si ya se ha insertado seguimos con la seleccion.
					
				if(t_tasks == total_tasks - 2)
				//if(t_tasks == (N_SINTETIC_TASKS/2) - 1)
				{
					if(inserted_tasks_type[task_max_dth] != 1)
						idx_selected_task = getMaxTask(time_DTH_tasks, inserted_tasks_type, total_tasks); //task_max_dth;
						
				}
			}
		}

		//Si el numero de tareas DK es mayor que el numero de tareas DT
		if(getNumberTypeDominant(h_type_dominant, total_tasks, DT) 
		< getNumberTypeDominant(h_type_dominant, total_tasks, DK))
		{
			//cout << "\tDK > DT" << endl;
#if PRUEBA_KDTH_MIN
			int first_task;
#endif
			if(t_tasks == 0)
			{
				float time = 1000000;
				
				for(int t = 0; t < total_tasks; t++)
				{
					
							//if(time_HTD_tasks[t] < time
							if(htd_time_heuristic[t] < time  
							&& inserted_tasks_type[t] != 1
							&& h_type_dominant[t] == DK)
							{
								//time = time_HTD_tasks[t];
								time = htd_time_heuristic[t];
								idx_selected_task = t;
								//inserted_type = DK;
						
						
							}
					
					
				}
#if PRUEBA_KDTH_MIN
				first_task = idx_selected_task;
#endif

			}
			else
			{
				//Si no es la tarea primera
				//Buscamos tareas DK que time_counter_HTD + HTD, no sobrepase
				//a time_counter_HTD + time_counter_K

#if PRUEBA_KDTH_MIN
				
				if(first_task != task_min_kdth)
				{
					inserted_tasks_type[task_min_kdth] = 1;
					if(t_tasks == total_tasks - 1)
						inserted_tasks_type[task_min_kdth] = 0;
				}
					
#endif
					
				int num_tasks = 0;
				bool flag = false;
				for(int t = 0; t < total_tasks; t++)
				{
					
						//if((time_counter_HTD + time_HTD_tasks[t] <= time_counter_K)
						if((time_counter_HTD + htd_time_heuristic[t] <= time_counter_K)  
						&& (inserted_tasks_type[t] != 1))
						{
							
							num_tasks++;
						}
						
					
				}

				if(num_tasks > 0)
				{
					//Existe mas de una tarea que time_counter_HTD + HTD, no sobrepasa
					//a time_counter_HTD + time_counter_K
						
					
					//Guardamos su tipo
					int *type_non_overlap = new int[num_tasks];
					int i = 0;
					for(int t = 0; t < total_tasks; t++)
					{
					
					
							if((time_counter_HTD + htd_time_heuristic[t] <= time_counter_K)  
							&& (inserted_tasks_type[t] != 1))
							{
								
								type_non_overlap[i] = t;
								i++;
							}
						
					
					}
						
						
					float time = 1000000;
						
					//De las tareas cuya HTD cuyo time_counter_HTD + HTD, no sobrepasa
					//a time_counter_HTD + time_counter_K, vemos aquellas cuya insercion
					//genera un tiempo menor
						
					//cout << "T_task: " << t_tasks << endl;
					//cout << "\t" << endl;
					for(int i = 0; i < num_tasks; i++)
					{
							
							//De las tareas DK cuyo time_counter_HTD + time_HTD no sobrepasa al time_counter_K
							//Seleccionamos aquella cuya diferencia time_counter_K - time_counter_HTD + time_HTD
							//sea menor
							if(time > (time_counter_K - (time_counter_HTD + htd_time_heuristic[type_non_overlap[i]])))
							{
								
								time = time_counter_K - (time_counter_HTD + htd_time_heuristic[type_non_overlap[i]]);
								idx_selected_task = type_non_overlap[i];
								flag = true;
								
							}
							
							
							
					}
						
						
					
					if(flag == false)
					{
							int max = -1;
							
							for(i = 0; i < num_tasks; i++)
							{
								
								if(time_DTH_tasks[type_non_overlap[i]] > max)
								{
									idx_selected_task = type_non_overlap[i];
									max = time_DTH_tasks[type_non_overlap[i]];
									
								}
								
							}
							
							
					}
							
					delete [] type_non_overlap;

				}
				else
				{
					//Vemos aquella tarea cuya insercion produca un tiempo menor
					float time = 10000000;
					int index;
						
						
					//Si para todas las tareas por insertar la suma time_counter_HTD + HTD
					//sobrepasa la suma time_counter_HTD + time_counter_K, seleccionamos
					//aquella tarea cuya diferencia (time_counter_HTD + HTD) - time_counter_K
					//sea menor
						
					for(int t = 0; t < total_tasks; t++)
					{
							//if((time > time_counter_HTD + time_HTD_tasks[t] - time_counter_K)
							if((time > time_counter_HTD + htd_time_heuristic[t] - time_counter_K)
							&& (inserted_tasks_type[t] != 1))
							{
								//time = time_counter_HTD + time_HTD_tasks[t] - time_counter_K;
								time = time_counter_HTD + htd_time_heuristic[t] - time_counter_K;
								idx_selected_task = t;
								
							}
							
							
					}
				}

				if(t_tasks == total_tasks - 2)
				//if(t_tasks == (N_SINTETIC_TASKS/2) - 1)
				{
						if(inserted_tasks_type[task_max_dth] != 1)
							//type = task_max_dth;
							idx_selected_task = getMaxTask(time_DTH_tasks, inserted_tasks_type, total_tasks);
						
				}
			}
		}

		//Para probar con el simulador un orden determinado.
		//idx_selected_task = order_tasks[id_epoch*N_SINTETIC_TASKS + t_tasks];
		
		inserted_tasks_type[idx_selected_task]++;

		
		int id = 0;
		int num_tasks = 0;
		
		for(int inserted_tasks = 0; inserted_tasks <= t_tasks; inserted_tasks++)
		{
				
			//Insertamos las tareas previamente seleccionadas
			if(id < t_tasks)
			{

				HTD_command.id                = id;
				HTD_command.id_stream         = execute_batch[h_tasks_pattern[id]];
				HTD_command.id_epoch          = id_epoch;
				HTD_command.ready             = false;
				HTD_command.overlapped        = false;
				HTD_command.t_ini             = INF_TIME;
				HTD_command.t_fin             = INF_TIME;
				HTD_command.t_CPU_GPU         = time_HTD_tasks[h_tasks_pattern[id]]; 
				HTD_command.enqueue           = false;
				HTD_command.launched          = false;
				HTD_command.t_overlap_CPU_GPU = time_overlapped_HTD_tasks[h_tasks_pattern[id]]; 

				kernel_command.id             = id;
				kernel_command.id_stream      = execute_batch[h_tasks_pattern[id]];
				kernel_command.id_epoch       = id_epoch;
				kernel_command.ready          = false;
				kernel_command.t_ini          = INF_TIME;
				kernel_command.t_fin          = INF_TIME;
				kernel_command.t_kernel       = time_kernel_tasks[h_tasks_pattern[id]];
				kernel_command.enqueue        = false;
				kernel_command.launched       = false;

				
				DTH_command.id                = id;
				DTH_command.id_stream         = execute_batch[h_tasks_pattern[id]];
				DTH_command.id_epoch          = id_epoch;
				DTH_command.ready             = false;
				DTH_command.overlapped        = false;
				DTH_command.enqueue           = false;
				DTH_command.active_htd        = false;
				DTH_command.t_ini             = INF_TIME;
				DTH_command.t_fin             = INF_TIME;
				DTH_command.t_GPU_CPU         =   time_DTH_tasks[h_tasks_pattern[id]]; 
				DTH_command.t_overlap_GPU_CPU = time_overlapped_DTH_tasks[h_tasks_pattern[id]];
				DTH_command.launched          = false;
					
					
			}
			else
			{
				
				
				HTD_command.id                = id;
				HTD_command.id_stream         = execute_batch[idx_selected_task];
				HTD_command.id_epoch          = id_epoch;
				HTD_command.ready             = false;
				HTD_command.overlapped        = false;
				HTD_command.t_ini             = INF_TIME;
				HTD_command.t_fin             = INF_TIME;
				HTD_command.t_CPU_GPU         =   time_HTD_tasks[idx_selected_task];
				HTD_command.enqueue           = false;
				HTD_command.launched          = false;
				HTD_command.t_overlap_CPU_GPU =  time_overlapped_HTD_tasks[idx_selected_task];

				kernel_command.id             = id;
				kernel_command.id_stream      = execute_batch[idx_selected_task];
				kernel_command.id_epoch       = id_epoch;
				kernel_command.ready          = false;
				kernel_command.t_ini          = INF_TIME;
				kernel_command.t_fin          = INF_TIME;
				kernel_command.t_kernel       = time_kernel_tasks[idx_selected_task]; 
				kernel_command.enqueue        = false;
				kernel_command.launched       = false;
			

				DTH_command.id                = id;
				DTH_command.id_stream         = execute_batch[idx_selected_task];
				DTH_command.id_epoch          = id_epoch;
				DTH_command.ready             = false;
				DTH_command.overlapped        = false;
				DTH_command.enqueue           = false;
				DTH_command.active_htd        = false;
				DTH_command.t_ini             = INF_TIME;
				DTH_command.t_fin             = INF_TIME;
				DTH_command.t_GPU_CPU         = time_DTH_tasks[idx_selected_task];
				DTH_command.t_overlap_GPU_CPU = time_overlapped_DTH_tasks[idx_selected_task];
				DTH_command.launched          = false;
			}
			
			deque_simulation_HTD.push_back(HTD_command);
				
			deque_simulation_K.push_back(kernel_command);
				
			deque_simulation_DTH.push_back(DTH_command);
				
			num_tasks++;
			id++;
			
		}
		
		
		//Declaramos unos punteros la inicio de los nuevos comandos en las colas.
		std::deque<infoCommand>::iterator current_K_begin = deque_simulation_K.begin() + num_K_enqueue;
		std::deque<infoCommand>::iterator current_HTD_begin = deque_simulation_HTD.begin() + num_HTD_enqueue;
		std::deque<infoCommand>::iterator current_DTH_begin = deque_simulation_DTH.begin() + num_DTH_enqueue;
		//Declaramos unos punteros para recorrer los nuevos comandos en las colas
		std::deque<infoCommand>::iterator current_K = deque_simulation_K.begin() + num_K_enqueue;
		std::deque<infoCommand>::iterator current_HTD = deque_simulation_HTD.begin() + num_HTD_enqueue;
		std::deque<infoCommand>::iterator current_DTH = deque_simulation_DTH.begin() + num_DTH_enqueue;

		
		
		//Recorremos la cola de simulacion HTD desde los comandos nuevos  hasta el final
		for(deque<infoCommand>::iterator it_HTD = current_HTD; 
			it_HTD != deque_simulation_HTD.end();
			it_HTD++) 
		{
			
			it_HTD->ready = true;
			
			//Recorremos la cola de simulacion DTH desde el comienzo hasta los comandos nuevos
			//(Solo recorremos las DTH de epocas anteriores)
			for(deque<infoCommand>::iterator current_DTH = deque_simulation_DTH.begin(); 
				current_DTH != current_DTH_begin; current_DTH++)
			{
				

				if(current_DTH->id_stream == it_HTD->id_stream)
				{
					//Si existe un DTH perteneciente al stream del HTD
					
					it_HTD->ready = false;
					current_DTH->next_command = it_HTD;
					current_DTH->active_htd = true;
					
				}
			
			
			}
			
			
			
		}
	

		
		//Establecemos las dependencias entre los nuevos comandos introducidos
		//HTD->K
		
		for(current_HTD = current_HTD_begin; current_HTD != deque_simulation_HTD.end(); current_HTD++)
		{
			current_HTD->next_command = current_K;
			current_K++;
			
		}
		
		//K->DTH
		current_DTH = current_DTH_begin;

		for(current_K = current_K_begin; current_K != deque_simulation_K.end(); current_K++)
		{
			
			current_K->next_command = current_DTH;
			current_DTH++;
		}
	
	
	
		//Si NO hay ningun comando en las colas de ejecucion inicializamos time_counter a 0
		//de lo contrario lo inicializaremos al tiempo de inicio menor entre los comandos de las
		//colas de ejecucion
		if(deque_execution_HTD.size() == 0 
		&& deque_execution_K.size() == 0 
		&& deque_execution_DTH.size() == 0)
			time_counter = 0;
		else
		{
			if(deque_execution_HTD.size() != 0)
				time_counter = deque_execution_HTD.begin()->t_ini;
			
			if(time_counter > deque_execution_K.begin()->t_ini && deque_execution_K.size() != 0)
				time_counter = deque_execution_K.begin()->t_ini;
			
			if(time_counter > deque_execution_DTH.begin()->t_ini && deque_execution_DTH.size() != 0)
				time_counter = deque_execution_DTH.begin()->t_ini;
			
		}

		
		//Empezamos a simular
		int i = 0;
	
		while(!deque_simulation_HTD.empty() || !deque_simulation_K.empty() || !deque_simulation_DTH.empty())
		{

			if(!deque_simulation_HTD.empty())
				current_HTD = deque_simulation_HTD.begin();
			if(!deque_simulation_K.empty())
				current_K = deque_simulation_K.begin();
			if(!deque_simulation_DTH.empty())
				current_DTH = deque_simulation_DTH.begin();

			//Si es un comando de la epoca actual, (t_ini == INF_TIME) inicializamos sus tiempos de inicio y fin
			if(!deque_simulation_HTD.empty() && current_HTD->t_ini == INF_TIME && current_HTD->ready == true)
			{
				current_HTD->t_ini = time_counter;
					
				current_HTD->t_fin = current_HTD->t_ini + current_HTD->t_CPU_GPU;
				current_HTD->t_estimated_fin = current_HTD->t_ini + current_HTD->t_CPU_GPU;
				
				
			}

			if(!deque_simulation_K.empty() && current_K->t_ini == INF_TIME && current_K->ready == true)
			{
				current_K->t_ini = time_counter;
				current_K->t_fin = current_K->t_ini + current_K->t_kernel;
				
				
			}

			if(!deque_simulation_DTH.empty() && current_DTH->t_ini == INF_TIME && current_DTH->ready == true)
			{
				current_DTH->t_ini = time_counter;
				current_DTH->t_fin = current_DTH->t_ini + current_DTH->t_GPU_CPU;
				current_DTH->t_estimated_fin = current_DTH->t_ini + current_DTH->t_GPU_CPU;
			
			
			}




			if(!deque_simulation_HTD.empty() && !deque_simulation_DTH.empty()
			&& current_HTD->t_ini != INF_TIME && current_DTH->t_ini != INF_TIME
			&& current_HTD->overlapped == false && current_DTH->overlapped == false
			&& current_HTD->ready == true && current_DTH->ready == true
			&& (current_HTD->launched == false || current_DTH->launched == false))
			{
				
				//Version 2 solapamiento
				float time_overlap_CPU_GPU;
				float time_overlap_GPU_CPU;
				
				float t_CPU_GPU = current_HTD->t_CPU_GPU;
				float t_overlap_CPU_GPU = current_HTD->t_overlap_CPU_GPU;
				float t_GPU_CPU = current_DTH->t_GPU_CPU;
				float t_overlap_GPU_CPU = current_DTH->t_overlap_GPU_CPU;
				
				
				//Solapamiento HTD version 2
				if(current_HTD->t_ini == time_counter)
				{
					
					
					if(current_DTH->t_fin < current_HTD->t_fin)	//Solapamos solo una parte de la transferencia HTD.
					{
									
									
						
						time_overlap_CPU_GPU = ((current_DTH->t_fin - time_counter)*t_overlap_CPU_GPU)/t_CPU_GPU;
								
						time_overlap_CPU_GPU = time_overlap_CPU_GPU + (current_HTD->t_fin - current_DTH->t_fin);
									
					}
					else
					{	
								
						
						//time_overlap_CPU_GPU = overlapped_time_CPU_GPU;
						time_overlap_CPU_GPU = t_overlap_CPU_GPU;
					}
							
							
				}
				else
				{
							
					if(current_DTH->t_fin < current_HTD->t_fin) //Solapamos solo una parte de la transferencia HTD.
					{
							
							time_overlap_CPU_GPU = ((current_DTH->t_fin - time_counter)*t_overlap_CPU_GPU)/t_CPU_GPU;
								
							time_overlap_CPU_GPU = (time_counter - current_HTD->t_ini) + time_overlap_CPU_GPU + (current_HTD->t_fin - current_DTH->t_fin);
								
								
					}
					else
					{
									
								
							//time_overlap_CPU_GPU = ((current_HTD->t_fin - time_counter)*overlapped_time_CPU_GPU)/time_CPU_GPU;
							time_overlap_CPU_GPU = ((current_HTD->t_fin - time_counter)*t_overlap_CPU_GPU)/t_CPU_GPU;
								
							time_overlap_CPU_GPU  = (time_counter - current_HTD->t_ini) + time_overlap_CPU_GPU;
									
								
					}
							
							
				}
						
				//Solapamiento HTD version 2
				if(current_DTH->t_ini == time_counter)
				{
							
							
							if(current_HTD->t_fin < current_DTH->t_fin)	//Solapamos solo una parte de la transferencia HTD.
							{
								
									//time_overlap_GPU_CPU = ((current_HTD->t_fin - time_counter)*overlapped_time_GPU_CPU)/time_GPU_CPU;
									time_overlap_GPU_CPU = ((current_HTD->t_fin - time_counter)*t_overlap_GPU_CPU)/t_GPU_CPU;
								
									time_overlap_GPU_CPU = time_overlap_GPU_CPU + (current_DTH->t_fin - current_HTD->t_fin);
		
							}
							else
							{
								//time_overlap_GPU_CPU = overlapped_time_GPU_CPU;
								time_overlap_GPU_CPU = t_overlap_GPU_CPU;
							
							}
				}
				else
				{
							
							
							if(current_DTH->t_fin < current_HTD->t_fin) //Solapamos solo una parte de la transferencia HTD.
							{
									
									time_overlap_GPU_CPU = ((current_DTH->t_fin - time_counter)*t_overlap_GPU_CPU)/t_GPU_CPU;
									time_overlap_GPU_CPU = (time_counter - current_DTH->t_ini) + time_overlap_GPU_CPU;
									
								
								
							}
							else
							{
									
									//time_overlap_GPU_CPU = ((current_HTD->t_fin - time_counter)*overlapped_time_GPU_CPU[permutation[current_DTH->id]])/time_GPU_CPU[permutation[current_DTH->id_stream]];
									
									
									time_overlap_GPU_CPU = ((current_HTD->t_fin - time_counter)*t_overlap_GPU_CPU)/t_GPU_CPU;
									time_overlap_GPU_CPU  = (time_counter - current_DTH->t_ini) + time_overlap_GPU_CPU + (current_DTH->t_fin - current_HTD->t_fin);
									
							}
							
							
				}
				
				
				
				
				//Si no es la HTD inicial
				if(current_HTD->t_fin != 0)
					current_HTD->t_fin = current_HTD->t_ini + time_overlap_CPU_GPU;
				
						
				current_HTD->overlapped = true;
						
				//Si no es la DTH inicial cuya duracion es 0
				if(current_DTH->t_fin != 0)
					current_DTH->t_fin = current_DTH->t_ini + time_overlap_GPU_CPU;
						
				
				current_HTD->t_estimated_fin = current_HTD->t_ini + time_overlap_CPU_GPU;
				current_DTH->t_estimated_fin = current_DTH->t_ini + time_overlap_GPU_CPU;
				
				
				
				current_DTH->overlapped = true;
				
				
	
						
			}

			memset(time_queues, INF_TIME, 3*sizeof(float));
		
			//Cojemos como time_counter el tiempo final de las colas mas pequeño
			if(deque_simulation_HTD.empty())
				time_queues[0] = INF_TIME;
			else
				time_queues[0] = current_HTD->t_fin;
				
			if(deque_simulation_K.empty())
				time_queues[1] = INF_TIME;
			else
				time_queues[1] = current_K->t_fin;
				
			if(deque_simulation_DTH.empty())
				time_queues[2] = INF_TIME;
			else
				time_queues[2] = current_DTH->t_fin;

			time_counter = getMinTime(time_queues, 3);


#if PRINT_SIMULATOR_TRACE			
			printf("\nTime Queues:\n");
			printf("HTD: %f\n", time_queues[0]);
			printf("K: %f\n", time_queues[1]);
			printf("DTH: %f\n", time_queues[2]);
			printf("Time Counter; %f\n", time_counter);
#endif
			

			if(!deque_simulation_HTD.empty() && current_HTD->t_fin == time_counter)
			{

				//Si no es un HTD perteneciente a la epoca anterior, resolvemos sus dependencias habilitamos el kernel
				current_HTD->next_command->ready = true;
			
				//cout << "\tActivando Kernel: " << current_HTD->next_command->id_stream << " ID: " << current_HTD->next_command->id << endl;
				

				if(!deque_simulation_DTH.empty())
					current_DTH->overlapped = false;


				//if(current_HTD->id == 0 && current_HTD->launched == false)
				//Si es el ultimo comando HTD a simular, lo encolamos
				if(current_HTD->id == nstreams - 1 && current_HTD->launched == false)
				{
					

					*t_current_ini_htd = current_HTD->t_ini;
					*t_current_fin_htd = current_HTD->t_fin;
				
				
					//Desencolamos el HTD del stream 0 (tarea primera)
					//encolamos los comandos K y DTH con id_stream == -1
					//que serian los comandos de la epoca anterior que todavia
					//no se han ejecutado
					infoCommand HTD_command;
				
					//HTD
					HTD_command.id_stream = current_HTD->id_stream;
					HTD_command.id = -1;
					HTD_command.id_epoch = current_HTD->id_epoch;
					HTD_command.ready = current_HTD->ready;
					HTD_command.overlapped = current_HTD->overlapped;
					HTD_command.t_ini = current_HTD->t_ini;
					//Cuando queremos meter un comando en la cola current (posterior cola ejecucion)
					//su tiempo de fin tiene que ser el tiempo de fin estimado (t_estimated_fin), ya que este tiempo 
					//solo refleja los incrementos ocasionados por el solapamiento con comandos lanzados anteriormente
					HTD_command.t_fin = current_HTD->t_estimated_fin;
					HTD_command.t_estimated_fin = current_HTD->t_estimated_fin;
					HTD_command.enqueue = true;
					HTD_command.launched = true;
					HTD_command.t_CPU_GPU = current_HTD->t_CPU_GPU;
					HTD_command.t_overlap_CPU_GPU = current_HTD->t_overlap_CPU_GPU;


					deque_current_HTD.push_back(HTD_command);

					for(deque<infoCommand>::iterator it_K = deque_simulation_K.begin(); 
					it_K != deque_simulation_K.end();
					it_K++)
					{
						//Si quedan comandos de kernels por desencolar, activamos su flag de encolamiento
						//en las colas de CURRENT para tenerlos en cuenta en las proximas epocas
						it_K->enqueue = true;
		
					}
				
					//Activamos el flag de encolamiento para el kernel perteneciente a la actual HTD
					current_HTD->next_command->enqueue = true;
					
					//DTH
					for(deque<infoCommand>::iterator it_DTH = deque_simulation_DTH.begin(); 
						it_DTH != deque_simulation_DTH.end();
						it_DTH++)
					{
						//Si quedan comandos de DTH por desencolar, activamos su flag de encolamiento
						//en las colas de CURRENT para tenerlos en cuenta en las proximas epocas
							
							
							it_DTH->enqueue = true;
						
						
					}
					
					//Activamos el flag de encolamiento para la DTH
					current_HTD->next_command->next_command->enqueue = true;


				}

				

				//Si es una HTD de la epoca anterior, es posible que se haya incrementado al solaparse
				//con una DTH de la epoca actual, si es asi modificamos los tiempos de inicio y fin
				// de las HTD de la epoca anterior

				//Si es una HTD de la epoca anterior
				if(current_HTD->id == -1)
				{
					
					//Modificamos los tiempos de inicio de las HTD pertenecientes
					//a la epoca anterior
					float t_fin_htd = current_HTD->t_fin;

					for(deque<infoCommand>::iterator it_HTD = current_HTD + 1; 
						it_HTD != deque_simulation_HTD.end();
						it_HTD++)
					{
						if(it_HTD->id == -1 && it_HTD->t_ini < t_fin_htd)
						{
							float t_dur = it_HTD->t_fin - it_HTD->t_ini;

							it_HTD->t_ini = t_fin_htd;
							it_HTD->t_fin = it_HTD->t_ini + t_dur;

							
						}

						t_fin_htd = it_HTD->t_fin;
						
					}
				}

				time_counter_HTD = current_HTD->t_fin;
				//Desencolamos
				deque_simulation_HTD.pop_front();

				

			}

			if(!deque_simulation_K.empty() && current_K->t_fin == time_counter)
			{
				//Resolvemos sus dependencias
				current_K->next_command->ready = true;

				

				if(current_K->id == nstreams - 1 && current_K->launched == false)
				{
					
					*t_current_ini_kernel = current_K->t_ini;
					*t_current_fin_kernel = current_K->t_fin;
				}

				

				//Comprobamos si el flag de encolamiento en las colas CURRENT esta activo
				if(current_K->enqueue == true)
				{
					//Si el flag esta activo, debemos encolar este comando en las colas CURRENT (poster colas ejecucion)
					//para tenerlo en cuenta en las proximas epocas.
					infoCommand kernel_command;
				
					kernel_command.id_stream = current_K->id_stream;
					kernel_command.id        = -1;
					kernel_command.id_epoch  = current_K->id_epoch;
					kernel_command.ready     = current_K->ready;
					kernel_command.t_ini     = current_K->t_ini;
					kernel_command.t_fin     = current_K->t_fin;
					kernel_command.t_kernel  = current_K->t_kernel;
					kernel_command.launched  = true;

					deque_current_K.push_back(kernel_command);
				}

				

				time_counter_K = current_K->t_fin;

				//Desencolamos
				deque_simulation_K.pop_front();


			}

			if(!deque_simulation_DTH.empty() && current_DTH->t_fin == time_counter)
			{
				//Desencolamos
			
				//Reseteamos el flag de solapamiento de la actual HTD
				if(!deque_simulation_HTD.empty())
					current_HTD->overlapped = false;

				//Si es una DTH de la epoca actual (launched == false), igualamos el tiempo de fin del ultimo comando 
				//lanzado en el stream, al tiempo de fin del comando DTH actual
				if(current_DTH->launched == false)
					t_current_last_dth_stream[current_DTH->id_stream] = current_DTH->t_estimated_fin;
			
				
				if(current_DTH->id == nstreams - 1 && current_DTH->launched == false)
				{
					*t_current_ini_dth = current_DTH->t_ini;
					*t_current_fin_dth = current_DTH->t_estimated_fin; 
				
				}

				//Comprobamos si el flag de encolamiento en las colas CURRENT esta activo
				if(current_DTH->enqueue == true)
				{
					infoCommand DTH_command;
				
					DTH_command.id_stream         = current_DTH->id_stream;
					DTH_command.id                = -1;
					DTH_command.id_epoch          = current_DTH->id_epoch;
					DTH_command.ready             = current_DTH->ready;
					DTH_command.enqueue           = true;
					DTH_command.launched          = true;
					DTH_command.overlapped        = false; //current_DTH->overlapped;
					DTH_command.t_ini             = current_DTH->t_ini;
					//Cuando queremos meter un comando en la cola current (posterior cola ejecucion)
					//su tiempo de fin tiene que ser el tiempo de fin estimado (t_estimated_fin), ya que este tiempo 
					//solo refleja los incrementos ocasionados con comando lanzados anteriormente
					DTH_command.t_fin             = current_DTH->t_estimated_fin;
					DTH_command.t_estimated_fin   = current_DTH->t_estimated_fin;
					DTH_command.t_GPU_CPU         = current_DTH->t_GPU_CPU;
					DTH_command.t_overlap_GPU_CPU = current_DTH->t_overlap_GPU_CPU;
					

					deque_current_DTH.push_back(DTH_command);

				}

				//Comprobamos si el DTH tiene activo su flag de dependencia. Si tiene activo este flag
				//significa que tiene que activar a un HTD
				if(current_DTH->active_htd == true)
				{
				
					current_DTH->next_command->ready = true;
					current_DTH->active_htd = false;
				}

				//Si es una DTH de la epoca anterior, es posible que se haya incrementado al solaparse
				//con una HTD de la epoca actual, si es asi modificamos los tiempos de inicio y fin
				// de las DTH de la epoca anterior

				//Si es una DTH de la epoca anterior
				if(current_DTH->id == -1)
				{
					//Modificamos los tiempos de inicio de las DTH pertenecientes
					//a la epoca anterior
					float t_fin_dth = current_DTH->t_fin;

					for(deque<infoCommand>::iterator it_DTH = current_DTH + 1; 
						it_DTH != deque_simulation_DTH.end();
						it_DTH++)
					{
						if(it_DTH->id == -1 && it_DTH->t_ini < t_fin_dth)
						{
							float t_dur = it_DTH->t_fin - it_DTH->t_ini;

							it_DTH->t_ini = t_fin_dth;
							it_DTH->t_fin = it_DTH->t_ini + t_dur;

							
						}

						t_fin_dth = it_DTH->t_fin;
						
					}

				}

				time_counter_DTH = current_DTH->t_fin;

				deque_simulation_DTH.pop_front();
			}

			i++;
		}
		
		h_tasks_pattern[t_tasks] = idx_selected_task;

		for(int i = 0; i < total_tasks; i++)
		{

			if(t_current_last_dth_stream[execute_batch[i]] < time_counter_HTD)
				htd_time_heuristic[i] = time_HTD_tasks[i];
			else
				htd_time_heuristic[i] = t_current_last_dth_stream[execute_batch[i]] - time_counter_HTD + time_HTD_tasks[i];
				
			
		}

	}

	for(int i = 0; i < total_tasks; i++)
	{
			if(t_current_last_dth_stream[execute_batch[i]] < time_counter_HTD)
				t_current_last_dth_stream[execute_batch[i]] = time_counter_HTD;
	}

	for(int i = 0; i < total_tasks; i++)
		h_order_processes[i] = execute_batch[h_tasks_pattern[i]];

	//delete [] time_tasks;
	delete [] h_tasks_pattern;
	delete [] h_type_dominant;
	delete [] inserted_tasks_type;
	delete [] htd_time_heuristic;

	return time_counter;
}

/**
 * @brief      Main function,
 * @details    Main function.
 * @author 	   Bernabe Lopez Albelda.
 * @date       02/03/2018
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
	
	int heu;
	if (str_heuristic == "HEURISTIC")
		heu = 1;
	else{
		if (str_heuristic == "NEH")
			heu = 2;
	}
	
	float *elapsed_times = new float[nIter];	//Execution times
	
	for(int iter = 0; iter < nIter; iter++){
#if MEASURE_TIME
		struct timeval t1, t2;
		gettimeofday(&t1, NULL);
#endif
		
		//Scheduling batch.
		//Batch scheduling_batch(max_tam_batch);
		int scheduling_batch = N_TASKS;
		int nstreams = nproducer;
		
		int scheduled_tasks = 0;
		
		//Launching order vector.
		int *h_order_processes = new int [nstreams];
		//Processes order vector in the execute batch.
		int *execute_batch = new int [nstreams];

		float *h_time_kernels_tasks = new float[N_TASKS];
		float *h_time_kernels_tasks_execute = new float[nstreams];

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
		
		//get name server
		char hostname[50];
		gethostname(hostname, 50);
		
		//get name gpu
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, gpu);
		
		string name_matrixTasks_file = "Matrix-Tasks_bench" + to_string(benchmark) + "_uniform_"
										+ to_string(N_TASKS) + "p_i" + str_interval1 + "-" + str_interval2
										+ "-" + hostname + "-" + prop.name + ".txt";
		ifstream archivo_entrada(name_matrixTasks_file);
		string linea;
		int app = 0;
		
		while(getline(archivo_entrada, linea)) {
			vector<string> v(tokenize (linea, "\t"));
			estimated_time_HTD[app] = atof(v[1].c_str());
			h_time_kernels_tasks[app] = atof(v[2].c_str());
			estimated_time_DTH[app] = atof(v[3].c_str());
			estimated_overlapped_time_HTD[app] = atof(v[4].c_str());
			estimated_overlapped_time_DTH[app] = atof(v[5].c_str());
			app++;
		}
		
		archivo_entrada.close();
		
		float t_previous_ini_htd, t_previous_fin_htd;
		float t_previous_ini_kernel, t_previous_fin_kernel;
		float t_previous_ini_dth, t_previous_fin_dth;
		float t_previous_overlap_htd, t_previous_overlap_dth;
		
		float t_current_ini_htd, t_current_fin_htd;
		float t_current_ini_kernel, t_current_fin_kernel;
		float t_current_ini_dth, t_current_fin_dth;
		float t_current_overlap_htd, t_current_overlap_dth;
	  
		float *t_previous_last_dth_stream = new float[nstreams];
		memset(t_previous_last_dth_stream, 0, nstreams * sizeof(float));
		float *t_current_last_dth_stream = new float[nstreams];
		memset(t_current_last_dth_stream, 0, nstreams * sizeof(float));
		
		t_previous_ini_htd = 0;
		t_previous_fin_htd = 0;
		t_previous_ini_kernel = 0;
		t_previous_fin_kernel = 0;
		t_previous_ini_dth = 0;
		t_previous_fin_dth = 0;
	  
		for(int i = 0; i < nstreams; i++)
			t_previous_last_dth_stream[i] = 0;
			
		for(int app = 0; app < N_TASKS; app++){
			execute_batch[app]                                    = app;
			h_time_kernels_tasks_execute[app]                     = h_time_kernels_tasks[app];
			estimated_time_HTD_per_stream_execute[app]            = estimated_time_HTD[app];
			estimated_time_DTH_per_stream_execute[app]            = estimated_time_DTH[app];
			estimated_overlapped_time_HTD_per_stream_execute[app] = estimated_overlapped_time_HTD[app];
			estimated_overlapped_time_DTH_per_stream_execute[app] = estimated_overlapped_time_DTH[app];
		}
		
		float time_simulation = 0;
	 
		for(int epoch = 0; epoch < nepoch; epoch++){	
			//Launching Heuristic
			switch (heu){
				case 1:
					time_simulation = heuristic(h_order_processes, scheduling_batch, execute_batch,
												h_time_kernels_tasks_execute, estimated_time_HTD_per_stream_execute, 
												estimated_time_DTH_per_stream_execute, 
												estimated_overlapped_time_HTD_per_stream_execute,
												estimated_overlapped_time_DTH_per_stream_execute,
												scheduling_batch, 
												t_previous_ini_htd, t_previous_ini_kernel, t_previous_ini_dth,
												t_previous_fin_htd, t_previous_fin_kernel, t_previous_fin_dth,
												&t_current_ini_htd, &t_current_ini_kernel, &t_current_ini_dth,
												&t_current_fin_htd, &t_current_fin_kernel, &t_current_fin_dth,
												t_previous_last_dth_stream, t_current_last_dth_stream, scheduled_tasks/nstreams);
				break;
				case 2:
					time_simulation = neh_F3(h_order_processes, h_time_kernels_tasks_execute, estimated_time_HTD_per_stream_execute, 
											estimated_time_DTH_per_stream_execute, estimated_overlapped_time_HTD_per_stream_execute,
											estimated_overlapped_time_DTH_per_stream_execute, t_previous_ini_htd, t_previous_ini_kernel,
											t_previous_ini_dth, t_previous_fin_htd, t_previous_fin_kernel, t_previous_fin_dth, 
											t_previous_overlap_htd, t_previous_overlap_dth, &t_current_ini_htd, &t_current_ini_kernel, 
											&t_current_ini_dth, &t_current_fin_htd, &t_current_fin_kernel, &t_current_fin_dth, 
											&t_current_overlap_htd, &t_current_overlap_dth, t_previous_last_dth_stream, 
											t_current_last_dth_stream, scheduled_tasks/nstreams);
				break;
				default:
					cout << "Choose a correct heuristic:" << endl;
					cout << "HEURISTIC" << endl;
					cout << "NEH" << endl;
				break;
			}
			

			cerr << "Time Simulation: " << time_simulation << endl;
			
			string name_orderTasks_file = "Order-Tasks_bench" + to_string(benchmark) + "_uniform_"
										+ to_string(nepoch) + "e_"
										+ to_string(N_TASKS) + "p_i0-0-" + hostname + "-"
										+ prop.name + "-" + str_heuristic + ".txt";
			ofstream order_file(name_orderTasks_file);
			for(int app = 0; app < N_TASKS; app++)
				order_file << h_order_processes[app] << " ";
			order_file << endl;
			order_file << "Time Simulation: " << time_simulation << endl;
			order_file.close();
			
			t_previous_ini_htd    = t_current_ini_htd;
			t_previous_fin_htd    = t_current_fin_htd;
			t_previous_ini_kernel = t_current_ini_kernel;
			t_previous_fin_kernel = t_current_fin_kernel;
			t_previous_ini_dth    = t_current_ini_dth;
			t_previous_fin_dth    = t_current_fin_dth;

			for(int i = 0; i < nstreams; i++)
				t_previous_last_dth_stream[i] = t_current_last_dth_stream[i];
			
			//PASAMOS EL CONTENIDO FINAL DE LAS COLAS CURRENT A LAS COLAS DE EJECUCION
			deque_execution_HTD.clear();   deque_execution_HTD = deque_current_HTD; deque_current_HTD.clear();
			deque_execution_K.clear();	   deque_execution_K   = deque_current_K;   deque_current_K.clear();
			deque_execution_DTH.clear();   deque_execution_DTH = deque_current_DTH; deque_current_DTH.clear();
			
			scheduled_tasks += N_TASKS;
		}
	 
		delete [] h_order_processes;
		delete [] execute_batch;
	 
		delete [] estimated_time_HTD; 
		delete [] estimated_time_DTH; 
		delete [] estimated_overlapped_time_HTD; 
		delete [] estimated_overlapped_time_DTH;
		
		delete [] estimated_time_HTD_per_stream_execute; 
		delete [] estimated_time_DTH_per_stream_execute; 
		delete [] estimated_overlapped_time_HTD_per_stream_execute; 
		delete [] estimated_overlapped_time_DTH_per_stream_execute;

		delete [] h_time_kernels_tasks;
		delete [] h_time_kernels_tasks_execute;

		delete [] t_previous_last_dth_stream;
		delete [] t_current_last_dth_stream;
	
#if MEASURE_TIME
		gettimeofday(&t2, NULL);
		double timer = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);
		float elapsed_time = timer/1000.0;

		elapsed_times[iter] = elapsed_time;
#endif	
	}
	
#if MEASURE_TIME
	cout << "median: " << getMedianTimeG(elapsed_times, nIter) << endl;
#endif	
	
	delete [] elapsed_times;

	return 0;
}
