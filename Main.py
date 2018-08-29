from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD;
rank = comm.Get_rank();
size=comm.Get_size();

if (rank==0):
    a=np.zeros((4,4));
    a[rank,:]=np.ones(4)*(rank+1);
    data=np.empty(4);
    datar=[];
    for i in range(1,4):
        comm.Recv(datar,source=i,tag=0);
        print(datar);
    print(a);
else:
    data=np.ones(4)*(rank+1);
    comm.Send([data,rank],dest=0,tag=0);

    
    
    