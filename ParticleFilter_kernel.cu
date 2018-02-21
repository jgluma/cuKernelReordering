/**
 * @file BlackScholes_kernel.cu
 * @details This file describes the kernel and device functions for a BlackScholes task.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#ifndef _PARTICLEFILTER_KERNEL_H_
#define _PARTICLEFILTER_KERNEL_H_

/*****************************
* CUDA Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: Nparticles
*****************************/
__global__ void kernel(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, int Nparticles){
    int block_id = blockIdx.x;// + gridDim.x * blockIdx.y;
    int i = blockDim.x * block_id + threadIdx.x;
    
    if(i < Nparticles){
    
        int index = -1;
        int x;
        
        for(x = 0; x < Nparticles; x++){
            if(CDF[x] >= u[i]){
                index = x;
                break;
            }
        }
        if(index == -1){
            index = Nparticles-1;
        }
        
        xj[i] = arrayX[index];
        yj[i] = arrayY[index];
        
    }
}

#endif
