/**
 * @file vectorAdd.h
 * @details This file describes a class to implement a matrix multiplication task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#ifndef _MATRIXMULT_H_
#define _MATRIXMULT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "task.h"

using namespace std;


class MatrixMult : public Task
{
private:
	float valB;
	/**
	 * A Matrix in host pinned memory.
	 */
	float *h_A_MM;
	/**
	 * B Matrix in host pinned memory.
	 */ 
	float *h_B_MM;
	/**
	 * C Matrix in host pinned memory.
	 */ 
	float *h_C_MM;
	/**
	 * A Matrix in device memory.
	 */
	float *d_A_MM;
	/**
	 * B Matrix in device memory.
	 */ 
	float *d_B_MM;
	/**
	 * C Matrix in device memory.
	 */ 
	float *d_C_MM;
	/**
	 * A matrix width
	 */
	unsigned int uiWA_MM;
	/**
	 * A matrix height
	 */
	unsigned int uiHA_MM;
	/**
	 * B matrix width
	 */
	unsigned int uiWB_MM;
	/**
	 * B matrix height
	 */
	unsigned int uiHB_MM;
	/**
	 * C matrix width
	 */
	unsigned int uiWC_MM;
	/**
	 * C matrix height
	 */
	unsigned int uiHC_MM;
	/**
	 * A matrix elements
	 */
	unsigned int mem_size_A_MM;
	/**
	 * B matrix elements
	 */
	unsigned int mem_size_B_MM;
	/**
	 * C matrix elements
	 */
	unsigned int mem_size_C_MM;
	/**
	 * Threads per block
	 */
	int blockSize;

	/**
	 * @brief Init data.
	 * @details This funciton initializes the input data for a matrix.
	 * 
	 * @param data Matrix data.
	 * @param size Matrix size.
	 * @param val Value for initialization.
	 */
	void constantInit(float *data, int size, float val);

public:
	/**
	 * @brief Constructor for the MatrixMult class.
	 * @details Default constructor.
	 * @author Antonio Jose Lazaro Munoz.
	 * @date 10/02/2016
	 */
	MatrixMult(int *sizes);
	/**
	 * @brief Destroyer for the MatrixMult class.
	 * @details This function implements the destroyer for the MatrixMult class. This function
	 * free the host and device memory.
	 * @author Antonio Jose Lazaro Munoz.
	 * @data 20/02/2016
	 */
	~MatrixMult();
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