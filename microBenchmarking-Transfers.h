/**
 * @file microBenchmarking-Transfers.h
 * @details This file describes a class to implement the PCIe microbenchmarking in CUDA.
 * @author Antonio Jose Lazaro Munoz.
 * @date 11/11/2017
 */
#ifndef _MICROBENCHMARKING_TRANSFERS_H_
#define _MICROBENCHMARKING_TRANSFERS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

using namespace std;

/**
 * @class MicroBenchmarkingTransfers
 * @brief CUDA PCIe microbenchmarking
 * @details This class implements the CUDA PCIe microbenchmarking
 * @author Antonio Jose Lazaro Munoz.
 * @date 11/11/2017
 * 
 */
class MicroBenchmarkingTransfers
{
    private:
        /**
         * GPU id
         */
		int gpuId;
        /**
         * GPU name
         */
		string gpuName;
        /**
         * GPU CUDA properties
         */
        cudaDeviceProp device_properties;
        /**
         * CUDA streams for microbenchmarking
         */
        cudaStream_t *stream_benchmark;
        /**
         * Number of iterations
         */
        int nIter;

        /**
         * HTD transfers latency
         */
	    float LoHTD;
        /**
         * DTH transfers latency
         */
        float LoDTH;
        /**
         * HTD transfers bandwidth
         */
        float GHTD;
        /**
         * Bandwidth of the overlapped HTD transfers.
         */
        float overlappedGHTD;
        /**
         * DTH transfers bandwidth
         */
        float GDTH;
        /**
         * Bandwidth of the overlapped DTH transfers.
         */
        float overlappedGDTH;

        /**
         * Minimum size of bytes for the microbenchmarking
         */
        int min_size_Bytes;
        /**
         * Maximum size of the bytes for the microbenchmarking
         */
        int max_size_Bytes;
        /**
         * Increment of bytes for the microbenchmarking
         */
        int increment_bytes;
        /**
         * HTD transfers times
         */
        vector<float> v_time_htd;
        /**
         * DTH transfers times.
         */
        vector<float> v_time_dth;
        /**
         * Times of the overlapped HTD transfers
         */
        vector<float> v_time_ohtd;
        /**
         * Times of the overlapped DTH transfers
         */
        vector<float> v_time_odth;

        /**
        * @brief Compute LoHTD
        * @details Function to compute the latency of the HTD transfers
        * @author Antonio Jose Lazaro Munoz.
        * @date 11/11/2017
        */
        void computeLoHTD(void);
        /**
        * @brief Compute LoDTH
        * @details Function to compute the latency of the DTH transfers
        * @author Antonio Jose Lazaro Munoz.
        * @date 11/11/2017
        */
        void computeLoDTH(void);
        /**
        * @brief Compute GHTD
        * @details Function to compute the bandwidth of the HTD transfers
        * @author Antonio Jose Lazaro Munoz.
        * @date 11/11/2017
        */
        void computeGHTD(void);
        /**
        * @brief Compute GDTH
        * @details Function to compute the bandwidth of the DTH transfers
        * @author Antonio Jose Lazaro Munoz.
        * @date 11/11/2017
        */
        void computeGDTH(void);
        /**
        * @brief Compute Overlapped Bandwidth
        * @details Function to compute the bandwidth of both the HTD transfers and DTH transfers
        * @author Antonio Jose Lazaro Munoz.
        * @date 11/11/2017
        */
        void computeOverlappedG(void);
         /**
        * @brief Compute Median
        * @details Function to compute median of a set of values
        * @author Antonio Jose Lazaro Munoz.
        * @date 11/11/2017
        */
        float getMedian(vector<float> v);
    
  public:
    /**
     * @brief Constructor for the MicroBenchmarkingTransfers class.
     * @details This function implements the constructor for the MicroBenchmarkingTransfers class. This
     * function initializes the required variables for the CUDA microbenchmarking.
     * @author Antonio Jose Lazaro Munoz.
     * @date 11/11/2017
     * 
     * @param gpuid GPU id.
     * @param min_Bytes Minimum size of the bytes for the microbenchmarking.
     * @param max_Bytes Maximum size of the bytes for the microbenchmarking.
     */
   	    MicroBenchmarkingTransfers(int gpuid, int min_Bytes, int max_Bytes);
    /**
     * @brief Destroyer for the MicroBenchmarkingTransfers class.
     * @details This function implements the destroyer for the MicroBenchmarkingTransfers class. 
     * @author Antonio Jose Lazaro Munoz.
     * @date 11/11/2017
     */
   	    ~MicroBenchmarkingTransfers();

        /**
         * @brief Execute microbenchmarking
         * @details Function to execute the PCIe microbenchmarking in CUDA.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         */
        void execute(void);
        /**
         * @brief Get GPU Id.
         * @details Function to get the GPU id.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return GPU id.
         */
        int getGPUid(void){return gpuId;};
        /**
         * @brief Get GPU name.
         * @details Function to get the GPU id.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return GPU name
         */
        string getGPUName(void){ return gpuName;};
        /**
         * @brief Get LoHTD
         * @details Function to get the latency of the HTD transfers.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Latency of the HTD transfers.
         */
        float getLoHTD(void){return LoHTD;};
        /**
         * @brief Get LoDTH
         * @details Function to get the latency of the DTH transfers.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Latency of the DTH transfers.
         */
        float getLoDTH(void){return LoDTH;};
        /**
         * @brief Get overlappedGHTD
         * @details Function to get the bandwidth of the overlapped HTD transfers.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Bandwidth of the HTD transfers.
         */
        float getOverlappedGHTD(void){return overlappedGHTD;};
        /**
         * @brief Get overlappedGDTH
         * @details Function to get the bandwidth of the overlapped DTH transfers.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Bandwidth of the DTH transfers.
         */
        float getOverlappedGDTH(void){return overlappedGDTH;};
        /**
         * @brief Get GHTD
         * @details Function to get the bandwidth of the HTD transfers.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Bandwidth of the HTD transfers.
         */
        float getGHTD(void){return GHTD;};
        /**
         * @brief Get GDTH
         * @details Function to get the bandwidth of the DTH transfers.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Bandwidth of the DTH transfers.
         */
        float getGDTH(void){return GDTH;};
        /**
         * @brief Get times HTD transfers.
         * @details Function to get the real times of the HTD transfers used for microbenchmarking.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Real times vector of the HTD transfers.
         */
        vector<float> getTimesHTD(void){return v_time_htd;};
        /**
         * @brief Get times DTH transfers.
         * @details Function to get the real times of the DTH transfers used for microbenchmarking.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Real times vector of the DTH transfers.
         */
        vector<float> getTimesDTH(void){return v_time_dth;};
        /**
         * @brief Get times overlapped HTD transfers.
         * @details Function to get the real times of the overlapped HTD transfers used for microbenchmarking.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Real times vector of the overlapped HTD transfers.
         */
        vector<float> getTimesOverlappedHTD(void){return v_time_ohtd;};
        /**
         * @brief Get times overlapped DTH transfers.
         * @details Function to get the real times of the overlapped DTH transfers used for microbenchmarking.
         * @author Antonio Jose Lazaro Munoz.
         * @date 11/11/2017
         * @return Real times vector of the overlapped DTH transfers.
         */
        vector<float> getTimesOverlappedDTH(void){return v_time_odth;};

};


#endif