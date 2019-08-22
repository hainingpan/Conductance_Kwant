#!/bin/bash
mpirun -np 4 python Majorana_main.py vznum=256  vzstep=0.002 leadnum=2 wireLength=500
mpirun -np 4 python Majorana_main.py vznum=256  vzstep=0.002 leadnum=1 leadpos=1 wireLength=500
