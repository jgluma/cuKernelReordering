#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark
	for epoch in `seq 1 4`
	do
		echo "EPOCH: " $epoch
		cat results_benchmark_$benchmark\_4p_$epoch\e_i0-0-PRUEBA.txt
	done
done