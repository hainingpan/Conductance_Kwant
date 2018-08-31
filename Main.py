#from mpi4py import MPI
#import numpy as np
#
#comm = MPI.COMM_WORLD;
#rank = comm.Get_rank();
#size=comm.Get_size();
#
#
# sendbuf=np.ones(4)*(rank+1)**2;
#if (rank==0):
#    recvbuf=np.empty((size,4));        
#else:   
#    recvbuf=None;
#    
#comm.Gather(sendbuf,recvbuf,root=0)
#
#if (rank==0):
#    print(recvbuf)
#
#    
#    
#    
import sys
import re
def main():
    vars=len(sys.argv);
    dict={'a':1,'b':2,'c':3};
    for i in range(vars):
        nam=re.serach('(\w)+=',sys.argv[i+1]);
        nam.group(1);
        
    print('a=',a,'b=',b,'c=',c)
    
if __name__=="__main__":
	main()
