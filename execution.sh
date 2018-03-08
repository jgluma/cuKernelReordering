#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark
	for epoch in `seq 1 4`
	do
		echo "EPOCH: " $epoch
		./execution 0 4 $epoch 4 ./benchmarks4tasks/ ./tiempos-aleatorios/ $benchmark  uniform 0 0 15
	done
done