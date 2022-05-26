
# Uncommet to run each case

#good1
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0
#good1se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=3 
#good3
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0
#good3se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=3


# small1
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0  -muVar 0.4 -muVar_fn=randlist_muVar0.4L1.dat
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=3 -muVar 0.4 -muVar_fn=randlist_muVar0.4L1.dat

# small3
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0  -muVar 0.4 -muVar_fn=randlist_muVar0.4L3.dat
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=3 -muVar 0.4 -muVar_fn=randlist_muVar0.4L3.dat


# deltaVar1
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -DeltaVar 0.06 -random_fn=randlist_gapVar0.06L1.dat
# deltaVar1se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=3 -DeltaVar 0.06 -random_fn=randlist_gapVar0.06L1.dat

# # deltaVar3
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -DeltaVar 0.06 -random_fn=randlist_gapVar0.06L3.dat
# # deltaVar3se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=3 -DeltaVar 0.06 -random_fn=randlist_gapVar0.06L3.dat

#qd1
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -QD_peak=1.7 -QD_L=0.2 -QD 
#qd1se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1 -QD_peak=1.7 -QD_L=0.2 -QD 

#qd3
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -QD_peak=0.6 -QD_L=0.4 -QD 
#qd3se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1 -QD_peak=0.6 -QD_L=0.4 -QD 

#inhom1
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -pot_type=exp -pot_sigma=0.15 -pot_peak=1.4
#inhom1se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1 -pot_type=exp -pot_sigma=0.15 -pot_peak=1.4

#inhom3
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -pot_type=exp -pot_sigma=0.4 -pot_peak=1.2
# #inhom3se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1 -pot_type=exp -pot_sigma=0.4 -pot_peak=1.2

#muVar1
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -muVar 1 -muVar_fn=randlist_muVar1L1.dat
#muVar1se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1.2 -muVar 1 -muVar_fn=randlist_muVar1L1.dat

#muVar2
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -muVar 1 -muVar_fn=randlist_muVar1_L1_2.dat
#muVar2se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1.2 -muVar 1 -muVar_fn=randlist_muVar1_L1_2.dat

#muVar3
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -muVar 1 -muVar_fn=randlist_muVar1L3.dat
#muVar3se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1.2 -muVar 1 -muVar_fn=randlist_muVar1L3.dat

#gvar1
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -gVar 0.8 -random_fn=randlist_gVar0.8L1.dat
#gvar1se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1.2 -gVar 0.8 -random_fn=randlist_gVar0.8L1.dat

#gvar2
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -gVar 0.8 -random_fn=randlist_gVar0.8_L1_2.dat
#gvar2se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1.2 -gVar 0.8 -random_fn=randlist_gVar0.8_L1_2.dat

#gvar3
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -gVar 0.6 -random_fn=randlist_gVar0.6L3.dat
#gvar3se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1.2 -gVar 0.6 -random_fn=randlist_gVar0.6L3.dat

#qd4
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -QD_peak=1.7 -QD_peak_R=2.3 -QD_L=0.2 -QD_L_R=0.15 -QD 
#qd4se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1 -QD_peak=1.7 -QD_peak_R=2.3 -QD_L=0.2 -QD_L_R=0.15 -QD 

#inhom4
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -pot_type=exp2 -pot_sigma=0.15 -pot_peak=1.4 -pot_sigma_R=0.1 -pot_peak_R=1.8 -pot_peak_pos_R=1
#inhom4se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1 -pot_type=exp2 -pot_sigma=0.15 -pot_peak=1.4 -pot_sigma_R=0.1 -pot_peak_R=1.8 -pot_peak_pos_R=1


#muVar4
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -muVar 1 -muVar_fn=randlist_muVar1_L1_4.dat
#muVar4se
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1.2 -muVar 1 -muVar_fn=randlist_muVar1_L1_4.dat

## don't remember what are these below??
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=128 -y_num=301 -SE -cond -LDOS -barrier_E=0 -dissipation=0

# mpirun -np 64 python legacy/Majorana_main.py wireLength=300 muVar=3 muVarList=randlist.dat mu=1 leadNum=2 xNum=128 yNum=301 isSE=1 barrierE=5

#pristine
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -x=mu -SE 

#qd
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -Vz=0 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -x=mu -SE -QD_peak=3 -QD_L=0.15 -QD

#inhom
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -Vz=0 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -pot_type=exp -pot_sigma=0.15 -pot_peak=1.4 -x=mu

#muVar4
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -Vz=0 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -SE -cond -barrier_E=0 -dissipation=0 -muVar 3 -muVar_fn=vimp.txt -x=mu

#exp1
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=501 -SE -cond -barrier_E=0 -dissipation=0 -Vzc=1.2 -muVar 1 -muVar_fn=randlist.dat -y_min=-0.5 -y_max=0.5

# pristine.sh

# good3
# mpirun -np 64 python -m mpi4py.futures Majorana_main.py -L=3 -mu=1 -lead_num=1 -lead_pos=LR -x_num=256 -y_num=301 -cond -barrier_E=0 -dissipation=0 -dissipation=1e-10