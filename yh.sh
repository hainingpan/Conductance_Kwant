#!/bin/bash
mpirun -np 8 python Majorana_main.py delta0=0.9 wireLength=500 barrierE=1 vc=6.7 mu=3 dissipation=0.01 vz0=2.5 vzNum=256 vzStep=0.004 vBiasMin=-0.4 vBiasMax=0.4 leadNum=2
mpirun -np 8 python Majorana_main.py delta0=0.9 wireLength=500 barrierE=1 vc=6.7 mu=3 dissipation=0.01 vz0=2.5 vzNum=256 vzStep=0.004 vBiasMin=-0.4 vBiasMax=0.4 leadNum=1 leadPos=1
