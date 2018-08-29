from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD;
rank = comm.Get_rank();
size=comm.Get_size();

if (rank==0):
#    a=np.zeros((4,4));
    sendbuf=np.ones(4)*(rank+1)
    recvbuf=np.empty((size,4));        
else:
    sendbuf=np.ones(4)*(rank+1)**2;
    recvbuf=None;
    
comm.Gather(sendbuf,recvbuf,root=0)

if (rank==0):
    print(recvbuf)

    
    
    