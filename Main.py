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
#    a=1;b=2;c=3;
    if vars>1:        
        for i in range(1,vars):
            try:
                varname=re.search('(.)*(?=\=)',sys.argv[i]).group(0);
                varval=float(re.search('(?<=\=)(.)*',sys.argv[i]).group(0));
                dict[varname]=varval;
            except:
                print('Cannot parse the input parameters',sys.argv[i]);
                sys.exit(1);
                
    print('a=',dict['a'],'b=',dict['b'],'c=',dict['c']);
    
if __name__=="__main__":
	main()
