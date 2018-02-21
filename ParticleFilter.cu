/**
 * @file ParticleFilter.cu
 * @details This file describes the functions belonging to ParticleFilter class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "ParticleFilter.h"
#include "ParticleFilter_kernel.cu"

ParticleFilter::ParticleFilter(int x, int y, int fr, int n)
{
	
	IszX       = x;
    IszY       = y;
    Nfr        = fr;
    Nparticles = n;

    max_size   = IszX*IszY*Nfr;
    //original particle centroid
    xe         = roundDouble(IszY/2.0);
    ye         = roundDouble(IszX/2.0);
    radius     = 5;
    diameter   = radius*2 - 1;
    countOnes  = 0;

	
}


ParticleFilter::~ParticleFilter()
{
	//Free host memory
	if(seed       != NULL) delete [] seed;
    if(I          != NULL) delete [] I;
    if(disk       != NULL) delete [] disk;
    if(objxy      != NULL) delete [] objxy;
    if(weights    != NULL) delete [] weights;
    if(likelihood != NULL) delete [] likelihood;
    if(ind        != NULL) delete [] ind;

    if(arrayX     != NULL) cudaFreeHost(arrayX);
    if(arrayY     != NULL) cudaFreeHost(arrayY);
    if(xj         != NULL) cudaFreeHost(xj);
    if(yj         != NULL) cudaFreeHost(yj);
    if(CDF        != NULL) cudaFreeHost(CDF);
    if(u          != NULL) cudaFreeHost(u);

	//Free device memory
	if(arrayX_GPU != NULL) cudaFree(arrayX_GPU);
    if(arrayY_GPU != NULL) cudaFree(arrayY_GPU);
    if(xj_GPU     != NULL) cudaFree(xj_GPU);
    if(yj_GPU     != NULL) cudaFree(yj_GPU);
    if(CDF_GPU    != NULL) cudaFree(CDF_GPU);
    if(u_GPU      != NULL) cudaFree(u_GPU);

    	

}

void ParticleFilter::allocHostMemory(void)
{
		
    seed       = new int [Nparticles];
    I          = new int[IszX*IszY*Nfr];
    disk       = new int[diameter*diameter];
    weights    = new double[Nparticles];
    likelihood = new double[Nparticles];

    cudaMallocHost((void **)&arrayX, sizeof(double)*Nparticles);
    cudaMallocHost((void **)&arrayY, sizeof(double)*Nparticles);
    cudaMallocHost((void **)&xj, sizeof(double)*Nparticles);
    cudaMallocHost((void **)&yj, sizeof(double)*Nparticles);
    cudaMallocHost((void **)&CDF, sizeof(double)*Nparticles);
    cudaMallocHost((void **)&u, sizeof(double)*Nparticles);
	
	
}

void ParticleFilter::freeHostMemory(void)
{
	
	
    
    //Free host memory
	if(seed       != NULL) delete [] seed;
    if(I          != NULL) delete [] I;
    if(disk       != NULL) delete [] disk;
    if(objxy      != NULL) delete [] objxy;
    

    if(weights    != NULL) delete [] weights;
    if(likelihood != NULL) delete [] likelihood;
    if(ind        != NULL) delete [] ind;

    if(arrayX     != NULL) cudaFreeHost(arrayX);
    if(arrayY     != NULL) cudaFreeHost(arrayY);
    if(xj         != NULL) cudaFreeHost(xj);
    if(yj         != NULL) cudaFreeHost(yj);
    if(CDF        != NULL) cudaFreeHost(CDF);
    if(u          != NULL) cudaFreeHost(u);
    
    
    
	
	
}

void ParticleFilter::allocDeviceMemory(void)
{
	
	//CUDA memory allocation
    cudaMalloc((void **) &arrayX_GPU, sizeof(double)*Nparticles);
    cudaMalloc((void **) &arrayY_GPU, sizeof(double)*Nparticles);
    cudaMalloc((void **) &xj_GPU,     sizeof(double)*Nparticles);
    cudaMalloc((void **) &yj_GPU,     sizeof(double)*Nparticles);
    cudaMalloc((void **) &CDF_GPU,    sizeof(double)*Nparticles);
    cudaMalloc((void **) &u_GPU,      sizeof(double)*Nparticles);
	
	
}

void ParticleFilter::freeDeviceMemory(void)
{
	
	//Free device memory
	if(arrayX_GPU != NULL) cudaFree(arrayX_GPU);
    if(arrayY_GPU != NULL) cudaFree(arrayY_GPU);
    if(xj_GPU     != NULL) cudaFree(xj_GPU);
    if(yj_GPU     != NULL) cudaFree(yj_GPU);
    if(CDF_GPU    != NULL) cudaFree(CDF_GPU);
    if(u_GPU      != NULL) cudaFree(u_GPU);

}

void ParticleFilter::generatingData(void)
{
	
	for(int i = 0; i < Nparticles; i++)
        seed[i] = time(NULL)*i;

    //call video sequence
    videoSequence(I, IszX, IszY, Nfr, seed);

    countOnes = 0;
    for(int x = 0; x < diameter; x++){
        for(int y = 0; y < diameter; y++){
            if(disk[x*diameter + y] == 1)
                countOnes++;
        }
    }

    strelDisk(disk, radius);

    //objxy = new double [countOnes];
    objxy = new double [2000];

   
    getneighbors(disk, countOnes, objxy, radius);


    for(int x = 0; x < Nparticles; x++){
        weights[x] = 1/((double)(Nparticles));
    }

    //ind = new int[countOnes];
    ind = new int[2000];



    for(int x = 0; x < Nparticles; x++){
        arrayX[x] = xe;
        arrayY[x] = ye;
    }
   
    for(int x = 0; x < Nparticles; x++){
            arrayX[x] = arrayX[x] + 1.0 + 5.0*randn(seed, x);
            arrayY[x] = arrayY[x] - 2.0 + 2.0*randn(seed, x);
    }

    int k = 1;
    for(int x = 0; x < Nparticles; x++){
        
            //compute the likelihood: remember our assumption is that you know
            // foreground and the background image intensity distribution.
            // Notice that we consider here a likelihood ratio, instead of
            // p(z|x). It is possible in this case. why? a hometask for you.        
            //calc ind
            for(int y = 0; y < countOnes; y++){
                indX = roundDouble(arrayX[x]) + objxy[y*2 + 1];
                indY = roundDouble(arrayY[x]) + objxy[y*2];
                ind[y] = fabs(indX*IszY*Nfr + indY*Nfr + k);
                if(ind[y] >= max_size)
                    ind[y] = 0;
            }
            likelihood[x] = calcLikelihoodSum(I, ind, countOnes);
            likelihood[x] = likelihood[x]/countOnes;
    }

    // update & normalize weights
    // using equation (63) of Arulampalam Tutorial      
    for(int x = 0; x < Nparticles; x++){
            weights[x] = weights[x] * exp(likelihood[x]);
    }

    sumWeights = 0;  
    for(int x = 0; x < Nparticles; x++){
            sumWeights += weights[x];
    }

    for(int x = 0; x < Nparticles; x++){
                weights[x] = weights[x]/sumWeights;
    }

    xe = 0;
    ye = 0;
    // estimate the object location by expected values
    for(int x = 0; x < Nparticles; x++){
        xe += arrayX[x] * weights[x];
        ye += arrayY[x] * weights[x];
    }

    distance = sqrt( pow((double)(xe-(int)roundDouble(IszY/2.0)),2) 
        + pow((double)(ye-(int)roundDouble(IszX/2.0)),2) );

    CDF[0] = weights[0];
    for(int x = 1; x < Nparticles; x++){
        CDF[x] = weights[x] + CDF[x-1];
    }

    u1 = (1/((double)(Nparticles)))*randu(seed, 0);
    for(int x = 0; x < Nparticles; x++){
            u[x] = u1 + x/((double)(Nparticles));
    }


	
}

void ParticleFilter::memHostToDeviceAsync(cudaStream_t stream)
{
	
	cudaMemcpyAsync(arrayX_GPU, arrayX, sizeof(double)*Nparticles, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(arrayY_GPU, arrayY, sizeof(double)*Nparticles, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(xj_GPU,     xj,     sizeof(double)*Nparticles, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(yj_GPU,     yj,     sizeof(double)*Nparticles, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(CDF_GPU,    CDF,    sizeof(double)*Nparticles, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(u_GPU,      u,      sizeof(double)*Nparticles, cudaMemcpyHostToDevice, stream);
    
	
}

void ParticleFilter::memHostToDevice(void)
{

    
    cudaMemcpy(arrayX_GPU, arrayX, sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(arrayY_GPU, arrayY, sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(xj_GPU,     xj,     sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(yj_GPU,     yj,     sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(CDF_GPU,    CDF,    sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(u_GPU,      u,      sizeof(double)*Nparticles, cudaMemcpyHostToDevice);


}

void ParticleFilter::memDeviceToHostAsync(cudaStream_t stream)
{
	
	cudaMemcpyAsync(yj, yj_GPU, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(xj, xj_GPU, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost, stream);
    
	
}

void ParticleFilter::memDeviceToHost(void)
{
    cudaMemcpy(yj, yj_GPU, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost);
    cudaMemcpy(xj, xj_GPU, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost);
}

void ParticleFilter::launch_kernel_Async(cudaStream_t stream)
{
	
	//Set number of threads

    int num_blocks = ceil((double) Nparticles/(double) threads_per_block);

    kernel <<< num_blocks, threads_per_block, 0, stream >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, 
        u_GPU, xj_GPU, yj_GPU, Nparticles);
	
	
}

void ParticleFilter::launch_kernel(void)
{
	//Set number of threads
    int num_blocks = ceil((double) Nparticles/(double) threads_per_block);

    kernel <<< num_blocks, threads_per_block>>> (arrayX_GPU, arrayY_GPU, CDF_GPU, 
        u_GPU, xj_GPU, yj_GPU, Nparticles);
    
        
}

void ParticleFilter::checkResults(void)
{
	
    for(int x = 0; x < Nparticles; x++){
            //reassign arrayX and arrayY
            arrayX[x] = xj[x];
            arrayY[x] = yj[x];
            weights[x] = 1/((double)(Nparticles));
    }
    
	
	
}

void ParticleFilter::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = Nparticles * sizeof(double) * 6;
	
	
}

void ParticleFilter::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = Nparticles * sizeof(double) * 2;
	
	
}

void ParticleFilter::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
								float *estimated_overlapped_time_HTD, float *estimated_overlapped_time_DTH, 
								float LoHTD, float LoDTH, float GHTD, float GDTH, float overlappedGHTD, float overlappedGDTH)
{
	
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, gpu);

	int bytes_HTD;
	int bytes_DTH;

	getBytesHTD(&bytes_HTD);
	getBytesDTH(&bytes_DTH);
	
	
			
	*estimated_time_HTD = LoHTD + (bytes_HTD) * GHTD;
				
	*estimated_overlapped_time_HTD = 0.0;
		
	if(props.asyncEngineCount == 2)
		*estimated_overlapped_time_HTD = LoHTD + (bytes_HTD) * overlappedGHTD;
			
		
	*estimated_time_DTH = LoDTH + (bytes_DTH) * GDTH;
				
	*estimated_overlapped_time_DTH= 0.0;

		
	if(props.asyncEngineCount == 2)
		*estimated_overlapped_time_DTH= LoDTH + (bytes_DTH) * overlappedGDTH;

	
	
}

/** 
* Takes in a double and returns an integer that approximates to that double
* @return if the mantissa < .5 => return value < input value; else return value > input value
*/
double ParticleFilter::roundDouble(double value){
    int newValue = (int)(value);
    if(value - newValue < .5)
    return newValue;
    else
    return newValue++;
}

/**
* Fills a radius x radius matrix representing the disk
* @param disk The pointer to the disk to be made
* @param radius  The radius of the disk to be made
*/
void ParticleFilter::strelDisk(int * disk, int radius)
{
    int diameter = radius*2 - 1;
    int x, y;
    for(x = 0; x < diameter; x++){
        for(y = 0; y < diameter; y++){
            double distance = sqrt(pow((double)(x-radius+1),2) + pow((double)(y-radius+1),2));
            if(distance < radius)
            disk[x*diameter + y] = 1;
        }
    }
}

/**
* The synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
* @param I The video itself
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames of the video
* @param seed The seed array used for number generation
*/
void ParticleFilter::videoSequence(int * I, int IszX, int IszY, int Nfr, int * seed){
    int k;
    int max_size = IszX*IszY*Nfr;
    /*get object centers*/
    int x0 = (int)roundDouble(IszY/2.0);
    int y0 = (int)roundDouble(IszX/2.0);
    I[x0 *IszY *Nfr + y0 * Nfr  + 0] = 1;
    
    /*move point*/
    int xk, yk, pos;
    for(k = 1; k < Nfr; k++){
        xk = abs(x0 + (k-1));
        yk = abs(y0 - 2*(k-1));
        pos = yk * IszY * Nfr + xk *Nfr + k;
        if(pos >= max_size)
        pos = 0;
        I[pos] = 1;
    }
    
    /*dilate matrix*/
    int * newMatrix = (int *)malloc(sizeof(int)*IszX*IszY*Nfr);
    imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
    int x, y;
    for(x = 0; x < IszX; x++){
        for(y = 0; y < IszY; y++){
            for(k = 0; k < Nfr; k++){
                I[x*IszY*Nfr + y*Nfr + k] = newMatrix[x*IszY*Nfr + y*Nfr + k];
            }
        }
    }
    free(newMatrix);
    
    /*define background, add noise*/
    setIf(0, 100, I, &IszX, &IszY, &Nfr);
    setIf(1, 228, I, &IszX, &IszY, &Nfr);
    /*add noise*/
    addNoise(I, &IszX, &IszY, &Nfr, seed);
}

/**
* Dilates the target matrix using the radius as a guide
* @param matrix The reference matrix
* @param dimX The x dimension of the video
* @param dimY The y dimension of the video
* @param dimZ The z dimension of the video
* @param error The error radius to be dilated
* @param newMatrix The target matrix
*/
void ParticleFilter::imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error, int * newMatrix)
{
    int x, y, z;
    for(z = 0; z < dimZ; z++){
        for(x = 0; x < dimX; x++){
            for(y = 0; y < dimY; y++){
                if(matrix[x*dimY*dimZ + y*dimZ + z] == 1){
                    dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
                }
            }
        }
    }
}

/**
* Dilates the provided video
* @param matrix The video to be dilated
* @param posX The x location of the pixel to be dilated
* @param posY The y location of the pixel to be dilated
* @param poxZ The z location of the pixel to be dilated
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param error The error radius
*/
void ParticleFilter::dilate_matrix(int * matrix, int posX, int posY, int posZ, 
    int dimX, int dimY, int dimZ, int error)
{
    int startX = posX - error;
    while(startX < 0)
    startX++;
    int startY = posY - error;
    while(startY < 0)
    startY++;
    int endX = posX + error;
    while(endX > dimX)
    endX--;
    int endY = posY + error;
    while(endY > dimY)
    endY--;
    int x,y;
    for(x = startX; x < endX; x++){
        for(y = startY; y < endY; y++){
            double distance = sqrt( pow((double)(x-posX),2) + pow((double)(y-posY),2) );
            if(distance < error)
            matrix[x*dimY*dimZ + y*dimZ + posZ] = 1;
        }
    }
}

/**
* Set values of the 3D array to a newValue if that value is equal to the testValue
* @param testValue The value to be replaced
* @param newValue The value to replace testValue with
* @param array3D The image vector
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
*/
void ParticleFilter::setIf(int testValue, int newValue, int * array3D, int * dimX, int * dimY, int * dimZ){
    int x, y, z;
    for(x = 0; x < *dimX; x++){
        for(y = 0; y < *dimY; y++){
            for(z = 0; z < *dimZ; z++){
                if(array3D[x * *dimY * *dimZ+y * *dimZ + z] == testValue)
                array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
            }
        }
    }
}

/**
* Sets values of 3D matrix using randomly generated numbers from a normal distribution
* @param array3D The video to be modified
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param seed The seed array
*/
void ParticleFilter::addNoise(int * array3D, int * dimX, int * dimY, int * dimZ, int * seed){
    int x, y, z;
    for(x = 0; x < *dimX; x++){
        for(y = 0; y < *dimY; y++){
            for(z = 0; z < *dimZ; z++){
                array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (int)(5*randn(seed, 0));
            }
        }
    }
}

/**
* Fills a 2D array describing the offsets of the disk object
* @param se The disk object
* @param numOnes The number of ones in the disk
* @param neighbors The array that will contain the offsets
* @param radius The radius used for dilation
*/
void ParticleFilter::getneighbors(int * se, int numOnes, double * neighbors, int radius){
    int x, y;
    int neighY = 0;
    int center = radius - 1;
    int diameter = radius*2 -1;
    for(x = 0; x < diameter; x++){
        for(y = 0; y < diameter; y++){
            if(se[x*diameter + y]){
               
                neighbors[neighY*2] = (int)(y - center);
                neighbors[neighY*2 + 1] = (int)(x - center);
                neighY++;
            }
        }
    }
}

/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
double ParticleFilter::randn(int * seed, int index){
    /*Box-Muller algorithm*/
    double u = randu(seed, index);
    double v = randu(seed, index);
    double cosine = cos(2*PI*v);
    double rt = -2*log(u);
    return sqrt(rt)*cosine;
}

/**
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
double ParticleFilter::randu(int * seed, int index)
{
    int num = A*seed[index] + C;
    seed[index] = num % M;
    return fabs(seed[index]/((double) M));
}

/**
* Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* @param I The 3D matrix
* @param ind The current ind array
* @param numOnes The length of ind array
* @return A double representing the sum
*/
double ParticleFilter::calcLikelihoodSum(int * I, int * ind, int numOnes){
    double likelihoodSum = 0.0;
    int y;
    for(y = 0; y < numOnes; y++)
    likelihoodSum += (pow((double)(I[ind[y]] - 100),2) - pow((double)(I[ind[y]]-228),2))/50.0;
    return likelihoodSum;
}