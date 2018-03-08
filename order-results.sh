#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark
	for epoch in `seq 1 4`
	do
		echo "EPOCH: " $epoch
		cat Order-Tasks_bench$benchmark\_uniform_$epoch\e_4p_i0-0-mistral-Tesla\ K20c-$1\.txt
	done
done