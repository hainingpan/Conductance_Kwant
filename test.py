from mpi4py import MPI
import numpy as np
import Majorana_module as Maj


comm = MPI.COMM_WORLD;
rank = comm.Get_rank();
size=comm.Get_size();
#size=1;
#rank=0;
tot=512;

NS_dict = {'a':1,'mu':.2,'alpha_R':5, 'Delta_0':0.2,'wireLength':100, 'mu_lead':25.0, 'Nbarrier':2,'Ebarrier':10.0, 'gamma':0.001, 'QD':'no', 'VD':0.8, 'dotLength':20, 'SE':'no', 'Vz':0.0, 'voltage':0.0,'varymu':'no', 'lamd':0};


np.warnings.filterwarnings('ignore');
voltageMin = -0.3; voltageMax = 0.3; voltageNumber = 1001;
voltageRange = np.linspace(voltageMin, voltageMax, voltageNumber);

per=int(tot/size);
junction=Maj.NSjunction(NS_dict);
VzStep = 0.002;    
sendbuf=np.empty((per,voltageNumber));
for ii in range(per):    
    NS_dict['Vz'] = (ii+rank*per)*VzStep;
#    gFile = open('G_mu'+str(NS_dict['mu'])+'_L'+str(NS_dict['wireLength'])+'_Vz'+ str(int(NS_dict['Vz']/VzStep))+'.txt','w');
    
    for index in range(voltageNumber):
        voltage=voltageRange(index);
        NS_dict['voltage']=voltage;        
        sendbuf[ii,index]=Maj.conductance(NS_dict,junction);

if (rank==0):
    recvbuf=np.empty((tot,voltageNumber));
else:
    recvbuf=None;
comm.Gather(sendbuf,recvbuf,root=0);
np.savetxt('G_mu'+str(NS_dict['mu'])+'_L'+str(NS_dict['wireLength'])+'_Delta'+str(NS_dict['Delta_0'])+'_alpha_R'+str(NS_dict['alpha'])+'.txt',recvbuf);
      
#        gFile.write( str(Maj.conductance(NS_dict,junction)) + ',' );
#    gFile.write('\n');
#    gFile.close();
