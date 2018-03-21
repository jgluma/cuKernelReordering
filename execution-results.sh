#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark
	for epoch in `seq 1 4`
	do
		echo "EPOCH: " $epoch
		cat results_benchmark_$benchmark\_$2\p_$epoch\e_i0-0-$1\.txt
	done
done