/**
 * @file Needle.h
 * @details This file describes a class to implement a Needle task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#ifndef _NEEDLE_H_
#define _NEEDLE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include "task.h"

using namespace std;

/**
 * @class Needle
 * @brief Needle class.
 * @details This class implements a Needle task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 * 
 */
class Needle : public Task
{
private:
	int max_rows_input; 
	int max_cols_input; 
	int max_rows;
	int max_cols;
	int penalty;
    int *input_itemsets;
    int *output_itemsets;
    int *referrence;
	int *matrix_cuda;
	int *referrence_cuda;
	int size;

	const int LIMIT = -999;

	const int blosum62[24][24] = {
		{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
		{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
		{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
		{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
		{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
		{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
		{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
		{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
		{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
		{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
		{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
		{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
		{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
		{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
		{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
		{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
		{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
		{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
		{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
		{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
		{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
		{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
		{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
		{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
		};
	
public:
	/**
	 * @brief Default constructor for the Needle class.
	 * @details Default constructor.
	 * @author Antonio Jose Lazaro Munoz.
	 * @date 10/02/2016
	 */
	Needle(int cols, int rows, int p);
	/**
	 * @brief Destroyer for the Needle class.
	 * @details This function implements the destroyer for the Needle class. This function
	 * free the host and device memory.
	 * @author Antonio Jose Lazaro Munoz.
	 * @data 20/02/2016
	 */
	~Needle();
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
