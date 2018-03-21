#!/bin/sh

for benchmark in `seq 1 5`
do
	echo "BENCHMARK: " $benchmark 
	cat Matrix-Tasks_bench$benchmark\_uniform_$1\p_i0-0-mistral-Tesla\ K20c.txt
done