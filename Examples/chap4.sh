# Uncommet to run each case
DIR=$(dirname $PWD)
N_cores=8

# muVar1um Fig3
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -Vzc=1.2 -x_max=1.2 -dissipation=0.003 -muVar=1 -muVar_fn=randlist_100.dat -barrier_E=13.0
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -Vzc=1.2 -x_max=1.2 -dissipation=0.003 -muVar=1 -muVar_fn=randlist_100.dat -barrier_E=14.5
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -Vzc=1.2 -x_max=1.2 -dissipation=0.003 -muVar=1 -muVar_fn=randlist_100.dat -barrier_E=15.3

# mu_dep Fig4
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -Vz=0.8 -lead_num=1 -lead_pos=LR -x=mu -x_num=256 -y_num=301 -cond -SE -Vzc=1.2 -x_min=0.95 -x_max=1.08 -dissipation=0.003 -muVar=1 -muVar_fn=randlist_100.dat -barrier_E=14.5
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -Vz=0.9 -lead_num=1 -lead_pos=LR -x=mu -x_num=256 -y_num=301 -cond -SE -Vzc=1.2 -x_min=0.95 -x_max=1.08 -dissipation=0.003 -muVar=1 -muVar_fn=randlist_100.dat -barrier_E=14.5
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -Vz=1.0 -lead_num=1 -lead_pos=LR -x=mu -x_num=256 -y_num=301 -cond -SE -Vzc=1.2 -x_min=0.95 -x_max=1.08 -dissipation=0.003 -muVar=1 -muVar_fn=randlist_100.dat -barrier_E=14.5

# vg_dep Fig4
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -mu=1 -Vz=0.8 -lead_num=1 -lead_pos=LR -x=barrier_E -x_num=256 -y_num=301 -cond -SE -Vzc=1.2 -x_min=0. -x_max=20 -dissipation=0.003 -muVar=1 -muVar_fn=randlist_100.dat
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -mu=1 -Vz=0.9 -lead_num=1 -lead_pos=LR -x=barrier_E -x_num=256 -y_num=301 -cond -SE -Vzc=1.2 -x_min=0. -x_max=20 -dissipation=0.003 -muVar=1 -muVar_fn=randlist_100.dat
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -mu=1 -Vz=1.0 -lead_num=1 -lead_pos=LR -x=barrier_E -x_num=256 -y_num=301 -cond -SE -Vzc=1.2 -x_min=0. -x_max=20 -dissipation=0.003 -muVar=1 -muVar_fn=randlist_100.dat

# WF and energy fig 5
mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=1024 -y_num=1001 -energy -SE -x_max=1.2 -dissipation=0.00 -barrier_E=0.0 
mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=1024 -y_num=1001 -energy -SE -Vzc=1.2 -x_max=1.2 -dissipation=0.00 -muVar=1 -muVar_fn=randlist_100.dat -barrier_E=0.0 


# muVar0.4um Fig6
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=0.4 -mu=5 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -coupling_SC_SM=1.5 -Vzc=10 -x_max=2.56 -y_min=-3 -y_max=3 -dissipation=0.1 -muVar=20 -muVar_fn=randlist_40.dat -barrier_E=10.0 -Delta0=3

# mu_dep Fig7
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=0.4 -Vz=0 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -coupling_SC_SM=1.5 -Vzc=10 -x=mu -x_min=4.5 -x_max=5.5 -y_min=-3 -y_max=3 -dissipation=0.1 -muVar=20 -muVar_fn=randlist_40.dat -barrier_E=10.0 -Delta0=3 
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=0.4 -Vz=0.88 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -coupling_SC_SM=1.5 -Vzc=10 -x=mu -x_min=4.5 -x_max=5.5 -y_min=-3 -y_max=3 -dissipation=0.1 -muVar=20 -muVar_fn=randlist_40.dat -barrier_E=10.0 -Delta0=3 
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=0.4 -Vz=1.45 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -coupling_SC_SM=1.5 -Vzc=10 -x=mu -x_min=4.5 -x_max=5.5 -y_min=-3 -y_max=3 -dissipation=0.1 -muVar=20 -muVar_fn=randlist_40.dat -barrier_E=10.0 -Delta0=3 


# vg_dep Fig7
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=0.4 -mu=5 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -coupling_SC_SM=1.5 -Vzc=10 -x=barrier_E -x_min=0 -x_max=20 -y_min=-3 -y_max=3 -dissipation=0.1 -muVar=20 -muVar_fn=randlist_40.dat -Vz=0.0 -Delta0=3
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=0.4 -mu=5 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -coupling_SC_SM=1.5 -Vzc=10 -x=barrier_E -x_min=0 -x_max=20 -y_min=-3 -y_max=3 -dissipation=0.1 -muVar=20 -muVar_fn=randlist_40.dat -Vz=0.88 -Delta0=3
# mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=0.4 -mu=5 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -SE -coupling_SC_SM=1.5 -Vzc=10 -x=barrier_E -x_min=0 -x_max=20 -y_min=-3 -y_max=3 -dissipation=0.1 -muVar=20 -muVar_fn=randlist_40.dat -Vz=1.45 -Delta0=3

# WF and energy fig 8
mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=0.4 -mu=5 -lead_num=2 -lead_pos=LR -x_num=1024 -y_num=1001 -energy -SE -coupling_SC_SM=1.5 -Vzc=10 -x_max=8 -y_min=-3 -y_max=3 -dissipation=0.1 -muVar=20 -muVar_fn=randlist_40.dat -barrier_E=10.0 -Delta0=3
mpirun -np $N_cores python -m mpi4py.futures $DIR/Majorana_main.py -L=0.4 -mu=5 -lead_num=2 -lead_pos=LR -x_num=1024 -y_num=1001 -energy -SE -coupling_SC_SM=1.5 -x_max=8 -y_min=-3 -y_max=3 -dissipation=0.1 -barrier_E=10.0 -Delta0=3
