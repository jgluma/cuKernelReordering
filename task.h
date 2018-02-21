/**
 * @file task.h
 * @details This file describes a class to implement a general task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#ifndef _TASK_H_
#define _TASK_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

/**
 * @class Task
 * @brief General Task
 * @details This class represented to a general task
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 * 
 */
class Task
{

public:

	/**
	 * @brief Alloc host memory
	 * @details Virtual function to alloc host pinned memory
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void allocHostMemory(void){}
	/**
	 * @brief free host memory
	 * @details Virtual function to free host pinned memory
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void freeHostMemory(void){};
	/**
	 * @brief Alloc device memory
	 * @details Virtual function to alloc device memory
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void allocDeviceMemory(void){};
	/**
	 * @brief Free device memory
	 * @details Virtual function to free device memory
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void freeDeviceMemory(void){};
	/**
	 * @brief Generating data
	 * @details Virtual function to generate input data.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void generatingData(void){};
	/**
	 * @brief Asynchronous HTD memory transfer.
	 * @details Virtual function to asynchronously perfom HTD memory tranfers.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
 	 * 
 	 * @param stream CUDA stream to launch the memory transfer.
	 */
	virtual void memHostToDeviceAsync(cudaStream_t stream){};
	/**
	 * @brief Synchronous HTD memory transfer.
	 * @details Virtual function to synchronously perform HTD memory transfers.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void memHostToDevice(void){};
	/**
	 * @brief Asynchronous DTH memory transfer.
	 * @details Virtual function to asynchronously perform DTH memory tranfers.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
 	 * 
 	 * @param stream CUDA stream to launch the memory transfer.
	 */
	virtual void memDeviceToHostAsync(cudaStream_t stream){};
	/**
	 * @brief Synchronous DTH memory transfer.
	 * @details Virtual function to synchronously perform DTH memory tranfers.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void memDeviceToHost(void){};
	/**
	 * @brief Asynchronous kernel launching.
	 * @details Virtual function to asynchronously perform a launching of the kernel.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
 	 * 
 	 * @param stream CUDA stream to launch the kernel.
	 */
	virtual void launch_kernel_Async(cudaStream_t stream){};
	/**
	 * @brief Synchronous kernel launching.
	 * @details Virtual function to synchronous perform a launching of the kernel.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void launch_kernel(void){};
	/**
	 * @brief Check results.
	 * @details Virtual function to perform a checking of the results.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void checkResults(void){};
	/**
	 * @brief Get HTD bytes.
	 * @details Virtual function to get the amount of bytes involved in a HTD transfer.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void getBytesHTD(int *bytes_htd){};
	/**
	 * @brief Get DTH bytes.
	 * @details Virtual function to get the amount of bytes involved in a DTH transfer.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	virtual void getBytesDTH(int *bytes_dth){};
	/**
	 * @brief Get HTD bytes.
	 * @details Virtual function to get the amount of bytes involved in a HTD transfer.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 *
	 * @param gpu GPU id.
	 * @param estimated_time_HTD HTD time estimation.
	 * @param estimated_time_DTH DTH time estimation.
	 * @param estimated_overlapped_time_HTD Time estimation of a overlap HTD transfer.
	 * @param estimated_overlapped_time_DTH Time estimation of a overlap HTD transfer.
	 * @param LoHTD PCIe latency for the HTD memory transfers.
	 * @param LoDTH PCIe latency for the DTH memory transfers.
	 * @param GHTD PCIe bandwidth for the HTD memory transfers.
	 * @param GDTH PCIe bandwidth for the DTH memory transfers.
	 * @param overlappedGHTD PCIe bandwidth for the overlapped HTD memory transfers.
	 * @param overlappedGDTH PCIe bandwidth for the overlapped DTH memory transfers.
	 */
	virtual void getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
			float *estimated_overlapped_time_HTD, float *estimated_overlapped_time_DTH, 
			float LoHTD, float LoDTH, float GHTD, float GDTH, float overlappedGHTD, float overlappedGDTH){};

};

#endif
