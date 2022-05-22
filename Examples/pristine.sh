#!/bin/bash
# rm m*
# mpirun -np 32 python Majorana_main.py wireLength=300 mu=1.00 alpha_R=5 xMax=2.048 xNum=256 leadNum=2 Q=0 yNum=301 barrierE=5 dissipation=0.0001 isSE=1 muVarList=randlist.dat muVar=2
# mpirun -np 32 python Majorana_main.py wireLength=300 mu=1.00 alpha_R=5 xMax=2.048 xNum=256 leadNum=2 Q=0 yNum=301 barrierE=10 dissipation=0.0001 isSE=1 muVarList=randlist.dat muVar=2
# mpirun -np 32 python Majorana_main.py wireLength=300 mu=1.00 alpha_R=5 xMax=2.048 xNum=256 leadNum=2 Q=0 yNum=301 barrierE=15 dissipation=0.0001 isSE=1 muVarList=randlist.dat muVar=2
# mpirun -np 32 python Majorana_main.py wireLength=300 mu=1.00 alpha_R=5 xMax=2.048 xNum=256 leadNum=2 Q=0 yNum=301 barrierE=20 dissipation=0.0001 isSE=1 muVarList=randlist.dat muVar=2

# mpirun -np 32 python Majorana_main.py wireLength=300 mu=1.00 alpha_R=5 xMax=2.048 xNum=256 leadNum=2 Q=0 yNum=301 barrierE=5 dissipation=0.0000 isSE=1
# mpirun -np 32 python Majorana_main.py wireLength=300 mu=1.00 alpha_R=5 xMax=2.048 xNum=256 leadNum=2 Q=0 yNum=301 barrierE=10 dissipation=0.0000 isSE=1
# mpirun -np 32 python Majorana_main.py wireLength=300 mu=1.00 alpha_R=5 xMax=2.048 xNum=256 leadNum=2 Q=0 yNum=301 barrierE=15 dissipation=0.0000 isSE=1
# mpirun -np 32 python Majorana_main.py wireLength=300 mu=1.00 alpha_R=5 xMax=2.048 xNum=256 leadNum=2 Q=0 yNum=301 barrierE=20 dissipation=0.0000 isSE=1

# mpirun -np 32 python Majorana_main.py wireLength=300 mu=1.00 couplingSCSM=1 alpha_R=5 xMax=2.048 xNum=128 leadNum=2 yNum=51 barrierE=20 dissipation=0.0000 isSE=1 isS=1

mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -muVar=3 -muVar_fn=randlist_short.dat -mu=1 -lead_num=2 -lead_pos=LR -x_num=128 -y_num=101 -SE -cond -LDOS -barrier_E=5
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -muVar=3 -muVar_fn=randlist_short.dat -mu=1 -lead_num=2 -lead_pos=LR -x_num=128 -y_num=101 -SE -cond -LDOS -barrier_E=10
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -muVar=3 -muVar_fn=randlist.dat -mu=1 -lead_num=2 -lead_pos=LR -x_num=128 -y_num=101 -SE -cond -LDOS -barrier_E=5

# mpirun -np 64 python legacy/Majorana_main.py wireLength=300 muVar=3 muVarList=randlist.dat mu=1 leadNum=2 xNum=128 yNum=101 isSE=1 barrierE=5