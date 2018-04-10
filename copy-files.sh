#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark
	for epoch in `seq 1 4`
	do
		echo "EPOCH: " $epoch
		cp Order-Tasks_bench$benchmark\_uniform_$epoch\e_$1\p_i0-0-mistral-Tesla\ K20c-HEURISTIC.txt Order-Tasks_bench$benchmark\_uniform_$epoch\e_$1\p_i0-0-mistral-Tesla\ K20c-JOHNSON.txt
		cp Order-Tasks_bench$benchmark\_uniform_$epoch\e_$1\p_i0-0-mistral-Tesla\ K20c-HEURISTIC.txt Order-Tasks_bench$benchmark\_uniform_$epoch\e_$1\p_i0-0-mistral-Tesla\ K20c-SLOPE.txt
		cp Order-Tasks_bench$benchmark\_uniform_$epoch\e_$1\p_i0-0-mistral-Tesla\ K20c-HEURISTIC.txt Order-Tasks_bench$benchmark\_uniform_$epoch\e_$1\p_i0-0-mistral-Tesla\ K20c-BIP.txt
	done
done