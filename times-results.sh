#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark 
	cat Matrix-Tasks_bench$benchmark\_uniform_4p_i0-0-mistral-Tesla\ K20c-HEURISTICO.txt
done