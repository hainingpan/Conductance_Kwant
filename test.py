#from mpi4py import MPI
import numpy as np
import Majorana_module as Maj


#comm = MPI.COMM_WORLD;
#rank = comm.Get_rank();
#size=comm.Get_size();
tot=512;

NS_dict = {'alpha':5/2, 'Delta_0':0.2,'wireLength':1000,'t':25.0, 'mu_lead':25.0, 'Nbarrier':2,'Ebarrier':10.0, 'gamma':0.001, 'QD':'no', 'VD':0.8, 'dotLength':20, 'SE':'no', 'Vz':0.0, 'voltage':0.0,'varymu':'no', 'mu':.2,'lamd':0};


np.warnings.filterwarnings('ignore');
voltageMin = -0.3; voltageMax = 0.3; voltageNumber = 1001;
voltageRange = np.linspace(voltageMin, voltageMax, voltageNumber);

per=int(tot/size);
for ii in range(per):
    VzStep = 0.002; NS_dict['Vz'] = (ii+rank*per)*VzStep;
    gFile = open('G_mu'+str(NS_dict['mu'])+'_L'+str(NS_dict['wireLength'])+'_Vz'+ str(int(NS_dict['Vz']/VzStep))+'.txt','w');
    for voltage in voltageRange:
        NS_dict['voltage']=voltage;
        gFile.write( str(Maj.conductance(NS_dict)) + ',' );
    gFile.write('\n');
    gFile.close();
