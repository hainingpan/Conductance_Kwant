#!/bin/bash
for l in `seq 5 5 25`
do
	for v in `seq 0.5 0.1 2`
	do
		echo "($l,$v)"
		mpiexec -n 4 python -m mpi4py.futures Majorana_adaptive.py isQD=1 qdPeak=$v qdLength=$l wireLength=100 isSE=1 isMu=1 muMax=2
	done
done
