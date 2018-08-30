from mpi4py import MPI
import numpy as np
import Majorana_module as Maj


comm = MPI.COMM_WORLD;
rank = comm.Get_rank();
size=comm.Get_size();
#size=1;
#rank=0;
tot=512;

NS_dict = {'a':1,'mu':0.2,'alpha_R':2, 'Delta_0':0.2,'Delta_c':0.2,'epsilon':0,'wireLength':300, 'mu_lead':25.0, 'Nbarrier':2,'Ebarrier':10.0, 'gamma':0.000, 'QD':'no', 'VD':0.8, 'dotLength':20, 'SE':'no', 'Vz':0.0, 'voltage':0.0,'varymu':'no', 'lamd':0,'singleband':'yes'};


np.warnings.filterwarnings('ignore');
voltageMin = -0.3; voltageMax = 0.3; voltageNumber = 1001;
voltageRange = np.linspace(voltageMin, voltageMax, voltageNumber);

per=int(tot/size);
VzStep = 0.002;  
sendbuf=np.empty((per,voltageNumber));
for ii in range(per):    
    NS_dict['Vz'] = (ii+rank*per)*VzStep;
    junction=Maj.NSjunction(NS_dict);
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
    if (NS_dict['singleband']=='yes'):
        fn='mu'+str(NS_dict['mu'])+'Delta'+str(NS_dict['Delta_0'])+'alpha'+str(NS_dict['alpha_R'])+'L'+str(NS_dict['wireLength'])+'.dat';     
    else:
        fn='mu'+str(NS_dict['mu'])+'Delta'+str(NS_dict['Delta_0'])+'alpha'+str(NS_dict['alpha_R'])+'Deltac'+str(NS_dict['Delta_c'])+'epsilon'+str(NS_dict['epsilon'])+'L'+str(NS_dict['wireLength'])+'.dat';
    np.savetxt(fn,recvbuf);      
