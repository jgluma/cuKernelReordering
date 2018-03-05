#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark 
	./timesEstimations 0 4 40 4 ./benchmarks4tasks/ ./tiempos-aleatorios/ $benchmark  uniform 0 0 1000
done