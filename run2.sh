#!/bin/bash
mpirun -np 4 python Majorana_main.py vzNum=256  vzStep=0.008 leadNum=2 wireLength=100 qdPeak=1.1 isQD=1 dotLength=25 isSE=1 couplingSCSM=0.2 mu=1
mpirun -np 4 python Majorana_main.py vzNum=256  vzStep=0.008 leadNum=1 leadpos=1 wireLength=100 qdPeak=1.1 isQD=1 dotLength=25 isSE=1 couplingSCSM=0.2 mu=1
mpirun -np 4 python Majorana_main.py vzNum=256  vzStep=0.008 leadNum=2 wireLength=300 qdPeak=1.1 isQD=1 dotLength=20 isSE=1 couplingSCSM=0.2 mu=1
mpirun -np 4 python Majorana_main.py vzNum=256  vzStep=0.008 leadNum=1 leadpos=1 wireLength=300 qdPeak=1.1 isQD=1 dotLength=20 isSE=1 couplingSCSM=0.2 mu=1
