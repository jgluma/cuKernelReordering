#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark
	for epoch in `seq 1 4`
	do
		echo "EPOCH: " $epoch
		./execution 0 $2 $epoch $2 ./benchmarks$2\tasks/ ./tiempos-aleatorios/ $benchmark  uniform 0 0 15 $1
	done
done