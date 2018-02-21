/**
 * @file TaskTemporizer.cu
 * @details This file describes the functions belonging to TaskTemporizer class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "TaskTemporizer.h"

TaskTemporizer::TaskTemporizer()
{
	
	cudaEventCreate(&start_event);
  	cudaEventCreate(&stop_event);
	

}

void TaskTemporizer::setTaskInsertionTime()
{
	gettimeofday(&t1, NULL);
}

void TaskTemporizer::setTaskLaunchTime()
{
	gettimeofday(&t2, NULL);
}

TaskTemporizer::~TaskTemporizer()
{
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

}

void TaskTemporizer::insertStartEventGPU(cudaStream_t *stream)
{
	cudaEventRecord(start_event, *stream);
}

void TaskTemporizer::insertStopEventGPU(cudaStream_t *stream)
{
	cudaEventRecord(stop_event, *stream);
}

float TaskTemporizer::getCPUTime()
{
	double timer = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);
	float elapsed_time = timer/1000.0;

	return elapsed_time;
}

float TaskTemporizer::getGPUTime()
{
	float elapsed_time;
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	return elapsed_time;
}

float TaskTemporizer::getTotalTime()
{
	double timer = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);
	float elapsed_time_CPU = timer/1000.0;

	float elapsed_time_GPU;
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&elapsed_time_GPU, start_event, stop_event);

	float total_time = elapsed_time_CPU + elapsed_time_GPU;

	return total_time;

}
