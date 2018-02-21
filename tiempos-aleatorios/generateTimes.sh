#!/bin/sh

#We obtain parameters from command line bash
NPRODUCER=2
NEPOCH=40
DISTRIBUTION=1

for interval in 0
do
		echo `./tiempos $NPRODUCER $NEPOCH $DISTRIBUTION 0 $interval`
		echo "./tiempos $NPRODUCER $NEPOCH $DISTRIBUTION 0 $interval"

done


