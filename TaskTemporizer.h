
#ifndef _TASKTEMPORIZER_H_
#define _TASKTEMPORIZER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>


using namespace std;


class TaskTemporizer
{
private:

	/**
	 * Start time of the tasks creation.
	 * 
	 */
	struct timeval t1;

	/**
	 * Time of the Task launching on GPU.
	 * 
	 */
	struct timeval t2;

	/**
	 * Start CUDA event of the task execution on GPU.
	 */
	cudaEvent_t start_event;

	/**
	 * Finish CUDA event of the task execution on GPU.
	 */
	cudaEvent_t stop_event;

public:
	/**
	 * @brief Constructor for the SinteticTask class.
	 * @details This function implements the constructor for the SinteticTask class. This
	 * function initializes the required variables for this task.
	 * @author Antonio Jose Lazaro Munoz.
	 * @date 20/02/2016
	 * 
	 * @param htd HTD percentage.
	 * @param computation Computation percentage.
	 * @param dth DTH percentage.
	 */
	TaskTemporizer();
	/**
	 * @brief Destroyer for the SinteticTask class.
	 * @details This function implements the destroyer for the SinteticTask class. This function
	 * free the host and device memory.
	 * @author Antonio Jose Lazaro Munoz.
	 * @data 20/02/2016
	 */
	~TaskTemporizer();

	void setTaskInsertionTime(void);
	void setTaskLaunchTime(void);
	void insertStartEventGPU(cudaStream_t *stream);
	void insertStopEventGPU(cudaStream_t *stream);
	float getCPUTime();
	float getGPUTime();
	float getTotalTime();
	


};

#endif
