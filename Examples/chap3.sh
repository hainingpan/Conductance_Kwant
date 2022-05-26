# Uncommet to run each case


# #good3
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001


# # qd
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=5 -QD_peak=0.4 -QD_L=0.15 -QD -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001

# # inhom
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=5 -pot_type=exp -pot_sigma=0.45 -pot_peak=1 -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001

# #muVar1p5
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=1.5 -muVar_fn=randlist_1p5.dat

# #muVar3
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=3 -muVar_fn=randlist_3.dat

# #muVar5
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=5 -muVar_fn=randlist_5.dat

# #qd+muVar5
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=5 -muVar_fn=randlist_5qd.dat -QD -QD_peak=0.4 -QD_L=0.15


# # good 0.5
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=0.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001
# # good 0.5c
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=0.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -Vzc=3

# #muVar3s
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=3 -muVar_fn=randlist_3s.dat

# #muVar3s
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=3 -muVar_fn=randlist_3s.dat -Vzc=3

# # muVar2
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=2 -muVar_fn=randlist_2.dat

# # muVar2.5
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=2.5 -muVar_fn=randlist_2.5.dat

# # muVar3.5
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=3.5 -muVar_fn=randlist_3.5.dat

# # muVar4
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=4 -muVar_fn=randlist_4.dat

# # muVar4.5
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=4.5 -muVar_fn=randlist_4.5.dat

# # qd+muVar0.5
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=0.5 -muVar_fn=randlist_0.5qd.dat -QD -QD_peak=0.4 -QD_L=0.15

# # qd+muVar3
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=3 -muVar_fn=randlist_3qd.dat -QD -QD_peak=0.4 -QD_L=0.15

# #muVar1.5s
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=1.5 -muVar_fn=randlist_1.5s.dat

# #muVar2s
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=2 -muVar_fn=randlist_2s.dat

# #muVar2.5s
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=2.5 -muVar_fn=randlist_2.5s.dat

# #muVar3.5s
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=3.5 -muVar_fn=randlist_3.5s.dat

# #muVar4s
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=4 -muVar_fn=randlist_4s.dat

# #muVar4.5s
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=4.5 -muVar_fn=randlist_4.5s.dat

# #muVar5s
mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=.5 -mu=1 -lead_num=2 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=5 -SE -coupling_SC_SM=1 -x_max=2.56 -dissipation=0.001 -muVar=5 -muVar_fn=randlist_5s.dat