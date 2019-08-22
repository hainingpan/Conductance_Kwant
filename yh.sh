#!/bin/bash
mpirun -np 4 python Majorana_main.py Delta_0=0.9 wireLength=500 Ebarrier=1 Vc=6.7 mu=3 Gamma=0.01 vz0=2.5 vznum=256 vzstep=0.004 voltagemin=-0.4 voltagemax=0.4 leadnum=2
mpirun -np 4 python Majorana_main.py Delta_0=0.9 wireLength=500 Ebarrier=1 Vc=6.7 mu=3 Gamma=0.01 vz0=2.5 vznum=256 vzstep=0.004 voltagemin=-0.4 voltagemax=0.4 leadnum=1 leadpos=1
