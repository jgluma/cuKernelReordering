/**
 * @file histogram.h
 * @details This file describes a class to implement a histogram task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "task.h"

using namespace std;

/**
 * @class Histogram
 * @brief Histogram class.
 * @details This class implements a Histogram task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 * 
 */
class Histogram : public Task
{
private:
	/**
	 * Host int vector for the input data.
	 */
	unsigned int *h_DataA;
	/**
	 * Host char vector for the input data.
	 */
	unsigned char *hh_DataA;
	/**
	 * Host int vector for CPU results.
	 */
	unsigned int *h_histoCPU;
	/**
	 * Host int vector for GPU results.
	 */
	unsigned int *h_histoGPU;
	/**
	 * Device vector for the input data.
	 */
	unsigned char *d_DataA;
	/**
	 * Device vector for the output data.
	 */
	unsigned int *d_histo;
	/**
	 * Device vector to reset data.
	 */
	unsigned int *d_zero;
	/**
	 * Number of frames
	 */
	int frames;

	/**
	 * Frame width
	 */
	int DATA_W;
	/**
	 * Frame height
	 */
	int DATA_H;
	/**
	 * Frame size
	 */
	int DATA_SIZE = DATA_W * DATA_H;
	/**
	 * Frame size divided by 4
	 */
	int DATA_SIZE4;
	/**
	 * Frame size (int)
	 */
	int DATA_SIZE_INT;

	//Align a to nearest higher multiple of b
	/**
	 * @brief AlignUp
	 * @details Align a to nearest higher multiple of b
	 * 
	 * @param a Value to align.
	 * @param b Reference value.
	 * 
	 * @return aligned value.
	 */
	int iAlignUp(int a, int b);

	/**
	 * @brief CPU histogram
	 * @details This function performs the histogram in CPU.
	 * @author CUDA SDK.
	 * 
	 * @param h_Result Histogram results.
	 * @param h_Data Input data.
	 * @param dataN Elements of the input data.
	 */
	void histogram256CPU(unsigned int *h_Result, unsigned int *h_Data, int dataN);

public:
	/**
	 * @brief Constructor for the Histogram class.
	 * @details This function implements the constructor for the Histogram class. This
	 * function initializes the required variables for this task.
	 * @author Antonio Jose Lazaro Munoz.
	 * @date 20/02/2016
	 * 
	 * @param nframes Number of frames.
	 */
	Histogram(int nframes);
	/**
	 * @brief Destroyer for the Histogram class.
	 * @details This function implements the destroyer for the Histogram class. This function
	 * free the host and device memory.
	 * @author Antonio Jose Lazaro Munoz.
	 * @data 20/02/2016
	 */
	~Histogram();
	/**
	 * @brief Alloc host memory
	 * @details Function to alloc host pinned memory
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void allocHostMemory(void);
	/**
	 * @brief free host memory
	 * @details Function to free host pinned memory
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void freeHostMemory(void);
	/**
	 * @brief Alloc device memory
	 * @details Function to alloc device memory
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void allocDeviceMemory(void);
	/**
	 * @brief Free device memory
	 * @details Function to free device memory
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void freeDeviceMemory(void);
	/**
	 * @brief Generating data
	 * @details Function to generate input data.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void generatingData(void);
	/**
	 * @brief Asynchronous HTD memory transfer.
	 * @details Function to asynchronously perfom HTD memory tranfers.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
 	 * 
 	 * @param stream CUDA stream to launch the memory transfer.
	 */
	void memHostToDeviceAsync(cudaStream_t stream);
	/**
	 * @brief Synchronous HTD memory transfer.
	 * @details Function to synchronously perform HTD memory transfers.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void memHostToDevice(void);
	/**
	 * @brief Asynchronous DTH memory transfer.
	 * @details Function to asynchronously perform DTH memory tranfers.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
 	 * 
 	 * @param stream CUDA stream to launch the memory transfer.
	 */
	void memDeviceToHostAsync(cudaStream_t stream);
	/**
	 * @brief Synchronous DTH memory transfer.
	 * @details Function to synchronously perform DTH memory tranfers.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void memDeviceToHost(void);
	/**
	 * @brief Asynchronous kernel launching.
	 * @details Function to asynchronously perform a launching of the kernel.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
 	 * 
 	 * @param stream CUDA stream to launch the kernel.
	 */
	void launch_kernel_Async(cudaStream_t stream);
	/**
	 * @brief Synchronous kernel launching.
	 * @details Function to synchronous perform a launching of the kernel.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void launch_kernel(void);
	/**
	 * @brief Check results.
	 * @details Function to perform a checking of the results.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void checkResults(void);
	/**
	 * @brief Get HTD bytes.
	 * @details Function to get the amount of bytes involved in a HTD transfer.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void getBytesHTD(int *bytes_htd);
	/**
	 * @brief Get DTH bytes.
	 * @details Function to get the amount of bytes involved in a DTH transfer.
	 * @author Antonio Jose Lazaro Munoz.
 	 * @date 20/02/2016
	 */
	void getBytesDTH(int *bytes_dth);
	/**
	 * @brief Get HTD bytes.
	 * @details Function to get the amount of bytes involved in a HTD transfer.
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
	void getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
			float *estimated_overlapped_time_HTD, float *estimated_overlapped_time_DTH, 
			float LoHTD, float LoDTH, float GHTD, float GDTH, float overlappedGHTD, float overlappedGDTH);


};

#endif
