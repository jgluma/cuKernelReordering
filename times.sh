#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark 
	./timesEstimations 0 $1 40 $1 ./benchmarks$1\tasks/ ./tiempos-aleatorios/ $benchmark  uniform 0 0 1000
done