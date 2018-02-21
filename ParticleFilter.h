/**
 * @file ParticleFilter.h
 * @details This file describes a class to implement a ParticleFilter task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#ifndef _PARTICLEFILTER_H_
#define _PARTICLEFILTER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include "task.h"

using namespace std;

/**
 * @class ParticleFilter
 * @brief ParticleFilter class.
 * @details This class implements a ParticleFilter task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 * 
 */
class ParticleFilter : public Task
{
private:
	int IszX;
	int IszY;
	int Nfr;
	int Nparticles;
	int max_size;
	int indX;
	int indY;
	double xe;
	double ye;
	int radius;
	int diameter;
	int countOnes;
	double sumWeights;
	double distance;  
	double u1;
	double * objxy;
	double * weights;
	double * likelihood;
	double * arrayX;
	double * arrayY;
	double * xj;
	double * yj;
	double * CDF;
	double * u;
	int * ind;

	//GPU copies of arrays
	double * arrayX_GPU;
	double * arrayY_GPU;
	double * xj_GPU;
	double * yj_GPU;
	double * CDF_GPU;
	double * u_GPU;

	int * seed;
	//matrix
	int * I;
	int * disk;

	const int threads_per_block = 128;

	/**
	@var A value for LCG
	*/
	const int A = 1103515245;
	/**
	@var C value for LCG
	*/
	const int C = 12345;

	const float PI = 3.1415926535897932;

	const int BLOCK_X = 16;
	const int BLOCK_Y = 16;

	const long M = INT_MAX;

	double roundDouble(double value);
	void strelDisk(int * disk, int radius);
	void videoSequence(int * I, int IszX, int IszY, int Nfr, int * seed);
	void imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error, int * newMatrix);
	void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error);
	void setIf(int testValue, int newValue, int * array3D, int * dimX, int * dimY, int * dimZ);
	void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ, int * seed);
	void getneighbors(int * se, int numOnes, double * neighbors, int radius);
	double randn(int * seed, int index);
	double randu(int * seed, int index);
	double calcLikelihoodSum(int * I, int * ind, int numOnes);
    
	

public:
	/**
	 * @brief Default constructor for the ParticleFilter class.
	 * @details Default constructor.
	 * @author Antonio Jose Lazaro Munoz.
	 * @date 10/02/2016
	 */
	ParticleFilter(int x, int y, int fr, int n);
	/**
	 * @brief Destroyer for the ParticleFilter class.
	 * @details This function implements the destroyer for the ParticleFilter class. This function
	 * free the host and device memory.
	 * @author Antonio Jose Lazaro Munoz.
	 * @data 20/02/2016
	 */
	~ParticleFilter();
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
