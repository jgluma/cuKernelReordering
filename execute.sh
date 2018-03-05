#!/bin/sh

#We obtain parameters from command line bash
REP=15
GPU=0
PATH_TASKS_FILE='./benchmarks4tasks/'
PATH_TIMES_FILE='./tiempos-aleatorios/'
INTERVAL1=0
NPRODUCER=4
MAXTAMBATCH=4
NEPOCH=40
DISTRIBUTION='uniform'
export CUDA_DEVICE_MAX_CONNECTIONS=1
PERCENT=100
#echo "$CUDA_DEVICE_MAX_CONNECTIONS"
F='results-'
F=$F'GPU'$GPU'.txt'

for benchmark in `seq 5 5`
do
	for interval2 in 0
	do
		
		echo `./streamsModel-VCPP $GPU $NPRODUCER $NEPOCH $MAXTAMBATCH $PATH_TASKS_FILE $PATH_TIMES_FILE $benchmark  $DISTRIBUTION $INTERVAL1 $interval2 $REP >> $F`
		echo "./streamsModel-VCPP $GPU $NPRODUCER $NEPOCH $MAXTAMBATCH $PATH_TASKS_FILE $PATH_TIMES_FILE $benchmark  $DISTRIBUTION $INTERVAL1 $interval2 $REP >> $F"
	done
done


# ./streamsModel-VCPP 0 4 40 4 ./benchmarks4tasks/ ./tiempos-aleatorios/ 5 uniform 0 0 1000

# ./timesEstimations 0 4 40 4 ./benchmarks4tasks/ ./tiempos-aleatorios/ 5 uniform 0 0 1000

# ./heuristic 0 4 40 4 ./benchmarks4tasks/ ./tiempos-aleatorios/ 5 uniform 0 0 300