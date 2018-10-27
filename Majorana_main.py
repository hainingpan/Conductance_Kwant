import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
import numpy as np
import Majorana_module as Maj
import sys
import re
import matplotlib.pyplot as plt

def main():
    vars=len(sys.argv);
    NS_dict = {'TV':1,'a':1,'mu':.2,'mumax':1,'alpha_R':5, 'Delta_0':0.2,'Delta_c':0.2,'epsilon':1,'wireLength':3000, 'mu_lead':25.0, 'Nbarrier':2,'Ebarrier':10.0, 'Gamma':0.0001, 'QD':'no', 'VD':0.4, 'dotLength':20, 'SE':'no', 'Vz':0.0, 'voltage':0.0,'smoothpot':0, 'gamma':0,'multiband':0,'leadpos':0,'peakpos':.25,'sigma':1};
    if vars>1:        
        for i in range(1,vars):
            try:
                varname=re.search('(.)*(?=\=)',sys.argv[i]).group(0);
                varval=re.search('(?<=\=)(.)*',sys.argv[i]).group(0);
                if varname in NS_dict:
                    if varname=='smoothpot':
                        NS_dict[varname]=varval;
                    else:
                        NS_dict[varname]=float(varval);
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
    tot=256;  
    if (rank==0):
        print(NS_dict);    
        
    np.warnings.filterwarnings('ignore');
    voltageMin = -.3; voltageMax = .3; voltageNumber = 1001;
    voltageRange = np.linspace(voltageMin, voltageMax, voltageNumber);
    
    per=int(tot/size);
    VzStep = 0.002;  
    sendbuf=np.empty((per,voltageNumber));  #conductance
    if NS_dict['TV']==1:
        sendbuf2=np.empty((per,voltageNumber)); #TV

    for ii in range(per):
        NS_dict['Vz'] = (ii+rank*per)*VzStep;
        junction=Maj.NSjunction(NS_dict);   #Change this if junction is voltage dependent, e.g. in Self energy
        for index in range(voltageNumber):
            voltage=voltageRange[index];
            NS_dict['voltage']=voltage;
            if NS_dict['TV']==0:
                sendbuf[ii,index]=Maj.conductance(NS_dict,junction);
            else:
                sendbuf[ii,index],sendbuf2[ii,index]=Maj.ConductanceAndTV(NS_dict,junction);
#            sendbuf2[ii,index]=Maj.TVmap(NS_dict,junction);    
    if (rank==0):
        recvbuf=np.empty((tot,voltageNumber)); 
        if NS_dict['TV']==1:
            recvbuf2=np.empty((tot,voltageNumber));
    else:
        recvbuf=None;
        if NS_dict['TV']==1:
            recvbuf2=None;
    comm.Gather(sendbuf,recvbuf,root=0);
    if NS_dict['TV']==1:
        comm.Gather(sendbuf2,recvbuf2,root=0);

    if (rank==0):
        fn_mu='m'+str(NS_dict['mu']);
        fn_Delta='D'+str(NS_dict['Delta_0']);
        fn_alpha='a'+str(NS_dict['alpha_R']);
        fn_wl='L'+str(int(NS_dict['wireLength']));
        fn_Deltac=('Dc'+str(NS_dict['Delta_c']))*(NS_dict['multiband']!=0);
        fn_epsilon=('ep'+str(NS_dict['epsilon']))*(NS_dict['multiband']!=0);
        fn_smoothpot=str(NS_dict['smoothpot'])*(NS_dict['smoothpot']!=0);
        fn_leadpos='L'*(NS_dict['leadpos']==0)+'R'*(NS_dict['leadpos']==1);
        fn_range='-'+str(VzStep*tot)+','+str(voltageMax)+'-';
        fn_mumax=('mx'+str(NS_dict['mumax']))*(NS_dict['smoothpot']!=0);
        fn_peakpos=('pk'+str(NS_dict['peakpos']))*(NS_dict['smoothpot']=='lorentz' or NS_dict['smoothpot']=='lorentzsigmoid');
        fn_sigma=('sg'+str(NS_dict['sigma']))*(NS_dict['smoothpot']=='exp');
        fn=fn_mu+fn_Delta+fn_alpha+fn_Deltac+fn_epsilon+fn_wl+fn_smoothpot+fn_mumax+fn_peakpos+fn_sigma+fn_leadpos+fn_range;
#        if (NS_dict['multiband']==0):
#            fn='mu'+str(NS_dict['mu'])+'Delta'+str(NS_dict['Delta_0'])+'alpha'+str(NS_dict['alpha_R'])+'L'+str(NS_dict['wireLength'])+str(NS_dict['smoothpot'])*(NS_dict['smoothpot']!=0)+'L'*(NS_dict['leadpos']==0)+'R'*(NS_dict['leadpos']==1)+'-'+str(VzStep*tot)+','+str(voltageMax)+'-.dat';     
#        else:
#            fn='mu'+str(NS_dict['mu'])+'Delta'+str(NS_dict['Delta_0'])+'alpha'+str(NS_dict['alpha_R'])+'Deltac'+str(NS_dict['Delta_c'])+'epsilon'+str(NS_dict['epsilon'])+'L'+str(NS_dict['wireLength'])+str(NS_dict['smoothpot'])*(NS_dict['smoothpot']!=0)+'-'+str(VzStep*tot)+','+str(voltageMax)+str(NS_dict['leadpos'])+'-.dat';
        np.savetxt(fn+'.dat',recvbuf);
        if NS_dict['TV']==1:
            np.savetxt(fn+'TV.dat',recvbuf2);

        magneticfieldrange=np.arange(tot)*VzStep;
        fig=plt.figure();
        plt.pcolormesh(magneticfieldrange,voltageRange,np.transpose(recvbuf));
        plt.xlabel('Vz(meV)');
        plt.ylabel('V_bias(meV)');
        plt.colorbar();
        plt.axis((0,tot*VzStep,voltageMin,voltageMax));
        fig.savefig(fn+'.png');
        
        if NS_dict['TV']==1:
            fig2=plt.figure();
            plt.pcolormesh(magneticfieldrange,voltageRange,np.transpose(recvbuf2));
            plt.xlabel('Vz(meV)');
            plt.ylabel('V_bias(meV)');
            plt.colorbar();
            plt.axis((0,tot*VzStep,voltageMin,voltageMax));
            fig2.savefig(fn+'TV.png');
    
if __name__=="__main__":
	main()

