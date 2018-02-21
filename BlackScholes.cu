/**
 * @file BlackScholes.cu
 * @details This file describes the functions belonging to BlackScholes class.
 * @author Antonio Jose Lazaro Munoz.
 * @date 20/02/2016
 */
#include "BlackScholes.h"
#include "BlackScholes_kernel.cu"

BlackScholes::BlackScholes(int op, int iterations)
{
	
	opt_n          = op;
	num_iterations = iterations;
	opt_sz = opt_n * sizeof(float);

	
}


BlackScholes::~BlackScholes()
{
	//Free host memory
	if(h_CallResultGPU	!=NULL) cudaFreeHost(h_CallResultGPU);
	if(h_PutResultGPU 	!=NULL) cudaFreeHost(h_PutResultGPU);
	if(h_StockPrice   	!=NULL) cudaFreeHost(h_StockPrice);
	if(h_OptionStrike 	!=NULL) cudaFreeHost(h_OptionStrike);
	if(h_OptionYears  	!=NULL) cudaFreeHost(h_OptionYears);

	if(h_CallResultCPU	!=NULL) delete [] h_CallResultCPU;
	if(h_PutResultCPU 	!=NULL) delete [] h_PutResultCPU;
    

	//Free device memory
	if(d_CallResult  	!=NULL) cudaFree(d_CallResult);
	if(d_PutResult   	!=NULL) cudaFree(d_PutResult);
	if(d_StockPrice  	!=NULL) cudaFree(d_StockPrice);
	if(d_OptionStrike	!=NULL) cudaFree(d_OptionStrike);
	if(d_OptionYears 	!=NULL) cudaFree(d_OptionYears);

    	

}

void BlackScholes::allocHostMemory(void)
{
		
    cudaMallocHost((void **)&h_CallResultGPU, opt_sz);
    cudaMallocHost((void **)&h_PutResultGPU, opt_sz);
    cudaMallocHost((void **)&h_StockPrice, opt_sz);
    cudaMallocHost((void **)&h_OptionStrike, opt_sz);
    cudaMallocHost((void **)&h_OptionYears, opt_sz);

    h_CallResultCPU = new float [opt_n];
    h_PutResultCPU  = new float [opt_n];
	
	
}

void BlackScholes::freeHostMemory(void)
{
	
	//Free host memory
	if(h_CallResultGPU	!=NULL) cudaFreeHost(h_CallResultGPU);
	if(h_PutResultGPU 	!=NULL) cudaFreeHost(h_PutResultGPU);
	if(h_StockPrice   	!=NULL) cudaFreeHost(h_StockPrice);
	if(h_OptionStrike 	!=NULL) cudaFreeHost(h_OptionStrike);
	if(h_OptionYears  	!=NULL) cudaFreeHost(h_OptionYears);

	if(h_CallResultCPU	!=NULL) delete [] h_CallResultCPU;
	if(h_PutResultCPU 	!=NULL) delete [] h_PutResultCPU;
	
}

void BlackScholes::allocDeviceMemory(void)
{
	
	cudaMalloc((void **)&d_CallResult,   opt_sz);
    cudaMalloc((void **)&d_PutResult,    opt_sz);
    cudaMalloc((void **)&d_StockPrice,   opt_sz);
    cudaMalloc((void **)&d_OptionStrike, opt_sz);
    cudaMalloc((void **)&d_OptionYears,  opt_sz);
	
	
}

void BlackScholes::freeDeviceMemory(void)
{
	
	//Free device memory
	if(d_CallResult  	!=NULL) cudaFree(d_CallResult);
	if(d_PutResult   	!=NULL) cudaFree(d_PutResult);
	if(d_StockPrice  	!=NULL) cudaFree(d_StockPrice);
	if(d_OptionStrike	!=NULL) cudaFree(d_OptionStrike);
	if(d_OptionYears 	!=NULL) cudaFree(d_OptionYears);

}

void BlackScholes::generatingData(void)
{
	
	srand(5347);

    //Generate options set
    for (int i = 0; i < opt_n; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }
	
}

void BlackScholes::memHostToDeviceAsync(cudaStream_t stream)
{
	
	 cudaMemcpyAsync(d_StockPrice,  h_StockPrice,   opt_sz, cudaMemcpyHostToDevice, stream);
     cudaMemcpyAsync(d_OptionStrike, h_OptionStrike,  opt_sz, cudaMemcpyHostToDevice, stream);
     cudaMemcpyAsync(d_OptionYears,  h_OptionYears,   opt_sz, cudaMemcpyHostToDevice, stream);
	
}

void BlackScholes::memHostToDevice(void)
{

     cudaMemcpy(d_StockPrice,  h_StockPrice,   opt_sz, cudaMemcpyHostToDevice);
     cudaMemcpy(d_OptionStrike, h_OptionStrike,  opt_sz, cudaMemcpyHostToDevice);
     cudaMemcpy(d_OptionYears,  h_OptionYears,   opt_sz, cudaMemcpyHostToDevice);


}

void BlackScholes::memDeviceToHostAsync(cudaStream_t stream)
{
	
	cudaMemcpyAsync(h_CallResultGPU, d_CallResult, opt_sz, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_PutResultGPU,  d_PutResult,  opt_sz, cudaMemcpyDeviceToHost, stream);
	
	
	
}

void BlackScholes::memDeviceToHost(void)
{
    cudaMemcpy(h_CallResultGPU, d_CallResult, opt_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  opt_sz, cudaMemcpyDeviceToHost);
}

void BlackScholes::launch_kernel_Async(cudaStream_t stream)
{
	
	for (int i = 0; i < num_iterations; i++)
    {
        BlackScholesGPU<<<DIV_UP((opt_n/2), 128), 128/*480, 128*/, 0, stream>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (float2 *)d_StockPrice,
            (float2 *)d_OptionStrike,
            (float2 *)d_OptionYears,
            RISKFREE,
            VOLATILITY,
            opt_n
        );
        
    }
	
	
}

void BlackScholes::launch_kernel(void)
{
	for (int i = 0; i < num_iterations; i++)
    {
        BlackScholesGPU<<<DIV_UP((opt_n/2), 128), 128/*480, 128*/>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (float2 *)d_StockPrice,
            (float2 *)d_OptionStrike,
            (float2 *)d_OptionYears,
            RISKFREE,
            VOLATILITY,
            opt_n
        );
        
    }
        


}

void BlackScholes::checkResults(void)
{
	
	//Calculate options values on CPU
    BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_StockPrice,
        h_OptionStrike,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        opt_n
    );

    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (int i = 0; i < opt_n; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
	
	
}

int BlackScholes::DIV_UP(int a, int b)
{
	return(( ((a) + (b) - 1) / (b) ));
}


void BlackScholes::getBytesHTD(int *bytes_htd)
{
	
	
	*bytes_htd = opt_sz * 3;
	
	
}

void BlackScholes::getBytesDTH(int *bytes_dth)
{
	
	*bytes_dth = opt_sz * 2;
	
	
}

void BlackScholes::getTimeEstimations_HTD_DTH(int gpu, float *estimated_time_HTD, float *estimated_time_DTH,
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

float BlackScholes::RandFloat(float low, float high)
{
    	float t = (float)rand() / (float)RAND_MAX;
    	return (1.0f - t) * low + t * high;
}

void BlackScholes::BlackScholesCPU(float *h_CallResult, float *h_PutResult, float *h_StockPrice,
    				float *h_OptionStrike, float *h_OptionYears, float Riskfree,
    				float Volatility, int optN)
{
	for (int opt = 0; opt < optN; opt++)
        BlackScholesBodyCPU(
            h_CallResult[opt],
            h_PutResult[opt],
            h_StockPrice[opt],
            h_OptionStrike[opt],
            h_OptionYears[opt],
            Riskfree,
            Volatility
        );
}

void BlackScholes::BlackScholesBodyCPU(float &callResult, float &putResult, float Sf, float Xf,
    						float Tf, float Rf, float Vf)
{
	 double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);
    callResult   = (float)(S * CNDD1 - X * expRT * CNDD2);
    putResult    = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}


double BlackScholes::CND(double d)
{
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}