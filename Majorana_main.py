import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
import numpy as np
import Majorana_module as Maj
import sys
import re
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD;
rank = comm.Get_rank();
size=comm.Get_size();
    
def main():    
    vars=len(sys.argv);    
    NS_dict = {'TV':0,'a':1,'mu':.2,'mumax':1,'alpha_R':5, 'Delta_0':0.2,'Delta_c':0.2,'epsilon':1,'wireLength':1000,
               'mu_lead':25.0, 'Nbarrier':2,'Ebarrier':10.0, 'Gamma':0.0001,'GammaVar':0, 
               'QD':0, 'VD':0.4, 'dotLength':20, 
               'SE':0, 'gamma':0.2, 'Vc':0,               
               'smoothpot':0, 'multiband':0,'leadpos':0,'peakpos':0,'sigma':1,
               'muVar':0,'muVarlist':0,
               'gVar':0,'randlist':0,
               'DeltaVar':0,
			   'alpha_RVar':0,
			   'massVar':0,
               'Vz':0.0, 'voltage':0.0,'vznum':256,'enum':1001,'vzstep': 0.002,'bothlead':0,
               'mustep':0.002,'munum':0,
               'Err':0};
    if (rank==0):
        if vars>1:        
            for i in range(1,vars):
                try:
                    varname=re.search('(.)*(?=\=)',sys.argv[i]).group(0);
                    varval=re.search('(?<=\=)(.)*',sys.argv[i]).group(0);
                    if varname in NS_dict:
                        if varname in ['smoothpot','muVarlist','randlist']:
                            NS_dict[varname]=varval;
                        else:
                            NS_dict[varname]=float(varval);
                    else:
                        print('Cannot find the parameter',varname);
                        NS_dict['Err']=1;
                        NS_dict=comm.bcast(NS_dict,root=0);
                        sys.exit(1);
                except:
                    print('Cannot parse the input parameters',sys.argv[i]);
                    NS_dict['Err']=1;
                    NS_dict=comm.bcast(NS_dict,root=0);
                    sys.exit(1);                   
        if (isinstance(NS_dict['muVarlist'],str)):
            print('disorder use filename:'+NS_dict['muVarlist']);
            muVarfn=NS_dict['muVarlist'];
            try:
                dat=np.loadtxt(muVarfn);
                try:
                    NS_dict['muVarlist']=dat;
                except:
                    print('Cannot read muVarlist',dat);
                    NS_dict['Err']=1;
                    NS_dict=comm.bcast(NS_dict,root=0);
                    sys.exit(1);
            except:
                print('Cannot find disorder file:',muVarfn);
                NS_dict['Err']=1;
                NS_dict=comm.bcast(NS_dict,root=0);
                sys.exit(1);
        else:                    
            if (NS_dict['muVar']!=0):
                NS_dict['muVarlist']=np.random.normal(0,NS_dict['muVar'],int(NS_dict['wireLength']));
        if (isinstance(NS_dict['randlist'],str)):
            print('randlist use filename:'+NS_dict['randlist']);
            randfn=NS_dict['randlist'];
            try:
                dat=np.loadtxt(randfn);
                try:
                    NS_dict['randlist']=dat;
                except:
                    print('Cannot read randlist',dat);
                    NS_dict['Err']=1;
                    NS_dict=comm.bcast(NS_dict,root=0);
                    sys.exit(1);
            except:
                print('Cannot find randlist file:',randfn);
                NS_dict['Err']=1;
                NS_dict=comm.bcast(NS_dict,root=0);                
                sys.exit(1);
        else:
            if (NS_dict['gVar']!=0):
                randlist=np.random.normal(1,NS_dict['gVar'],int(NS_dict['wireLength']));
                while not (np.prod(randlist>0)):
                    randlist=np.random.normal(1,NS_dict['gVar'],int(NS_dict['wireLength']));  
                NS_dict['randlist']=randlist;
                
            if (NS_dict['DeltaVar']!=0):
                randlist=np.random.normal(NS_dict['Delta_0'],NS_dict['DeltaVar'],int(NS_dict['wireLength']));
                while not (np.prod(randlist>0)):
                    randlist=np.random.normal(NS_dict['Delta_0'],NS_dict['DeltaVar'],int(NS_dict['wireLength']));
                NS_dict['randlist']=randlist;     

            if (NS_dict['alpha_RVar']!=0):
                randlist=np.random.normal(NS_dict['alpha_R'],NS_dict['alpha_RVar'],int(NS_dict['wireLength']));
                while not (np.prod(randlist>0)):
                    randlist=np.random.normal(NS_dict['alpha_R'],NS_dict['alpha_RVar'],int(NS_dict['wireLength']));
                NS_dict['randlist']=randlist; 				
			
            if (NS_dict['massVar']!=0):
                randlist=np.random.normal(1,NS_dict['massVar'],int(NS_dict['wireLength']));
                while not (np.prod(randlist>0)):
                    randlist=np.random.normal(1,NS_dict['massVar'],int(NS_dict['wireLength']));
                NS_dict['randlist']=randlist;   
				
			
                               
        
        print(NS_dict);   

        
    NS_dict=comm.bcast(NS_dict,root=0);
    if NS_dict['Err']!=0:   #for the slave to exit
#        print('I am rank=',rank,'My flag is',NS_dict['Err'],'I exit because',NS_dict['Err']!=0);
        sys.exit(1);
    if NS_dict['munum']==0:
        tot=int(NS_dict['vznum']);
        vzstep = NS_dict['vzstep'];  
    else:
        mu0=NS_dict['mu'];
        tot=int(NS_dict['munum']); 
        mustep=NS_dict['mustep'];      
    np.warnings.filterwarnings('ignore');
    voltageMin = -.3; voltageMax = .3; voltageNumber = int(NS_dict['enum']);
    voltageRange = np.linspace(voltageMin, voltageMax, voltageNumber);
    
    randlist=NS_dict['randlist'];
    
    per=int(tot/size);    
    
    for irun in range(int(NS_dict['bothlead'])+1):  #if both leads needed, running sequentially
        if NS_dict['bothlead']==1:
            NS_dict['leadpos']=irun;
        sendbuf=np.empty((per,voltageNumber));  #conductance
        if NS_dict['TV']==1:
            sendbuf2=np.empty((per,voltageNumber)); #TV
        
        for ii in range(per):
            if NS_dict['munum']==0:
                NS_dict['Vz'] = (ii+rank*per)*vzstep;
            else:
                NS_dict['mu'] = mu0+(ii+rank*per)*mustep;
                
            if NS_dict['gVar']!=0:
                NS_dict['randlist']=randlist*NS_dict['Vz'];
            if NS_dict['SE']==0:
                junction=Maj.NSjunction(NS_dict);   #Change this if junction is voltage dependent, e.g. in Self energy
            for index in range(voltageNumber):
                voltage=voltageRange[index];
                NS_dict['voltage']=voltage;
                if NS_dict['SE']==1:
                    junction=Maj.NSjunction(NS_dict);                    
                if NS_dict['TV']==0:
                    sendbuf[ii,index]=Maj.conductance(NS_dict,junction);
                else:
                    sendbuf[ii,index],sendbuf2[ii,index]=Maj.ConductanceAndTV(NS_dict,junction);
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
            if NS_dict['munum']==0:
                fn_range=('-'+str(vzstep*tot)+','+str(voltageMax)+'-')*(NS_dict['munum']==0);
            else:
                fn_range=('-'+str(mu0)+','+str(mu0+mustep*tot)+','+str(voltageMax)+'-')*(NS_dict['munum']!=0);
            fn_mumax=('mx'+str(NS_dict['mumax']))*(NS_dict['smoothpot']!=0);
            fn_peakpos=('pk'+str(NS_dict['peakpos']))*((NS_dict['smoothpot']=='lorentz')+( NS_dict['smoothpot']=='lorentzsigmoid'));
            fn_sigma=('sg'+str(NS_dict['sigma']))*((NS_dict['smoothpot']=='exp')+(NS_dict['smoothpot']=='sigmoid'));
            fn_muVar=('mVar'+str(NS_dict['muVar']))*(NS_dict['muVar']!=0)
            fn_Gamma=('G'+str(NS_dict['Gamma']))*(NS_dict['GammaVar']!=0);
            fn_dotLength=('dL'+str(int(NS_dict['dotLength'])))*(NS_dict['QD']!=0);
            fn_VD=('VD'+str(NS_dict['VD']))*(NS_dict['QD']!=0);
            fn_gamma=('g'+str(NS_dict['gamma']))*(NS_dict['SE']==1);
            fn_Vc=('Vc'+str(NS_dict['Vc']))*(NS_dict['SE']==1)*(NS_dict['Vc']!=0);
            fn_gVar=('gVar'+str(NS_dict['gVar']))*(NS_dict['gVar']!=0);
            fn_DeltaVar=('DVar'+str(NS_dict['DeltaVar']))*(NS_dict['DeltaVar']!=0);
            fn_alpha_RVar=('aVar'+str(NS_dict['alpha_RVar']))*(NS_dict['alpha_RVar']!=0);
            fn_tVar=('massVar'+str(NS_dict['massVar']))*(NS_dict['massVar']!=0);
            
            fn=fn_mu+fn_Delta+fn_DeltaVar+fn_alpha+fn_alpha_RVar+fn_Deltac+fn_epsilon+fn_wl+fn_smoothpot+fn_mumax+fn_peakpos+fn_sigma+fn_muVar+fn_VD+fn_dotLength+fn_gamma+fn_Vc+fn_Gamma+fn_gVar+fn_tVar+fn_leadpos+fn_range;
            
            np.savetxt(fn+'.dat',recvbuf);
            if NS_dict['TV']==1:
                np.savetxt(fn+'TV.dat',recvbuf2);
            if NS_dict['munum']==0:
                xrange=np.arange(tot)*vzstep;
            else:
                xrange=mu0+np.arange(tot)*mustep;
            fig=plt.figure();
            plt.pcolormesh(xrange,voltageRange,np.transpose(recvbuf), cmap='rainbow');
            if NS_dict['munum']==0:            
                plt.xlabel('Vz(meV)');
            else:
                plt.xlabel('mu(meV)');
            plt.ylabel('V_bias(meV)');
            plt.colorbar();
            plt.axis((xrange[0],xrange[-1],voltageMin,voltageMax));
            fig.savefig(fn+'.png');
            
            if NS_dict['TV']==1:
                fig2=plt.figure();
                plt.pcolormesh(xrange,voltageRange,np.transpose(recvbuf2));
                plt.xlabel('Vz(meV)');
                plt.ylabel('V_bias(meV)');
                plt.colorbar();
                plt.axis((0,tot*vzstep,voltageMin,voltageMax));
                fig2.savefig(fn+'TV.png');
        
   
if __name__=="__main__":
	main()

