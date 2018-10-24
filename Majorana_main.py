from mpi4py import MPI
import numpy as np
import Majorana_module as Maj
import sys
import re

def main():
    vars=len(sys.argv);
    NS_dict = {'a':1,'mu':.2,'alpha_R':2, 'Delta_0':0.2,'Delta_c':0.2,'epsilon':1,'wireLength':3000, 'mu_lead':25.0, 'Nbarrier':2,'Ebarrier':10.0, 'Gamma':0.0001, 'QD':'no', 'VD':0.8, 'dotLength':20, 'SE':'no', 'Vz':0.0, 'voltage':0.0,'varymu':'no', 'lamd':0,'multiband':0};
    if vars>1:        
        for i in range(1,vars):
            try:
                varname=re.search('(.)*(?=\=)',sys.argv[i]).group(0);
                varval=float(re.search('(?<=\=)(.)*',sys.argv[i]).group(0));
                if varname in NS_dict:
                    NS_dict[varname]=varval;
                else:
                    print('Cannot find the parameter',varname);
                    sys.exit(1);
            except:
                print('Cannot parse the input parameters',sys.argv[i]);
                sys.exit(1);    
      
    comm = MPI.COMM_WORLD;
    rank = comm.Get_rank();
    size=comm.Get_size();
#    size=1;
#    rank=0;
    tot=1024;  
    if (rank==0):
        print(NS_dict);    
        
    np.warnings.filterwarnings('ignore');
    voltageMin = -.3; voltageMax = .3; voltageNumber = 1001;
    voltageRange = np.linspace(voltageMin, voltageMax, voltageNumber);
    
    per=int(tot/size);
    VzStep = 0.002;  
    sendbuf=np.empty((per,voltageNumber));
    for ii in range(per):    
        NS_dict['Vz'] = (ii+rank*per)*VzStep;
        junction=Maj.NSjunction(NS_dict);   #Change this if junction is voltage dependent, e.g. in Self energy
        for index in range(voltageNumber):
            voltage=voltageRange[index];
            NS_dict['voltage']=voltage;        
            sendbuf[ii,index]=Maj.conductance(NS_dict,junction);
    if (rank==0):
        recvbuf=np.empty((tot,voltageNumber));
    else:
        recvbuf=None;
    comm.Gather(sendbuf,recvbuf,root=0);
    if (rank==0):
        if (NS_dict['multiband']==0):
            fn='mu'+str(NS_dict['mu'])+'Delta'+str(NS_dict['Delta_0'])+'alpha'+str(NS_dict['alpha_R'])+'L'+str(NS_dict['wireLength'])+'-'+str(VzStep*tot)+','+str(voltageMax)+'-.dat';     
        else:
            fn='mu'+str(NS_dict['mu'])+'Delta'+str(NS_dict['Delta_0'])+'alpha'+str(NS_dict['alpha_R'])+'Deltac'+str(NS_dict['Delta_c'])+'epsilon'+str(NS_dict['epsilon'])+'L'+str(NS_dict['wireLength'])+'-'+str(VzStep*tot)+','+str(voltageMax)+'-.dat';
        np.savetxt(fn,recvbuf);
        
    
if __name__=="__main__":
	main()

