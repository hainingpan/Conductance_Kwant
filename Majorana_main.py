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
    parameters = {'isTV':0,'a':1,'mu':.2,'muMax':1,'alpha_R':5, 'delta0':0.2,'wireLength':1000,
               'muLead':25.0, 'barrierNum':2,'barrierE':10.0, 'dissipation':0.0001,'isDissipationVar':0, 
               'isQD':0, 'qdPeak':0.4, 'dotLength':20, 
               'isSE':0, 'couplingSCSM':0.2, 'vc':0,               
               'potType':0,'potPeakPos':0,'potSigma':1,
               'muVar':0,'muVarList':0,
               'gVar':0,'randList':0,
               'deltaVar':0,
               'vz':0.0,'vz0':0, 'vBias':0.0,'vBiasMin':-0.3,'vBiasMax':0.3,'vzNum':256,'vBiasNum':1001,'vzStep': 0.002,
               'leadPos':0,'leadNum':1,
               'muStep':0.002,'muNum':0,
               'error':0};
    if (rank==0):
        if vars>1:        
            for i in range(1,vars):
                try:
                    varName=re.search('(.)*(?=\=)',sys.argv[i]).group(0);
                    varValue=re.search('(?<=\=)(.)*',sys.argv[i]).group(0);
                    if varName in parameters:
                        if varName in ['potType','muVarList','randList']:
                            parameters[varName]=varValue;
                        else:
                            parameters[varName]=float(varValue);
                    else:
                        print('Cannot find the parameter',varName);
                        parameters['error']=1;
                        parameters=comm.bcast(parameters,root=0);
                        sys.exit(1);
                except:
                    print('Cannot parse the input parameters',sys.argv[i]);
                    parameters['error']=1;
                    parameters=comm.bcast(parameters,root=0);
                    sys.exit(1);                   
        if (isinstance(parameters['muVarList'],str)):
            print('disorder use filename:'+parameters['muVarList']);
            muVarfn=parameters['muVarList'];
            try:
                dat=np.loadtxt(muVarfn);
                try:
                    parameters['muVarList']=dat;
                except:
                    print('Cannot read muVarList',dat);
                    parameters['error']=1;
                    parameters=comm.bcast(parameters,root=0);
                    sys.exit(1);
            except:
                print('Cannot find disorder file:',muVarfn);
                parameters['error']=1;
                parameters=comm.bcast(parameters,root=0);
                sys.exit(1);
        else:                    
            if (parameters['muVar']!=0):
                parameters['muVarList']=np.random.normal(0,parameters['muVar'],int(parameters['wireLength']));
        if (isinstance(parameters['randList'],str)):
            print('random list use filename:'+parameters['randList']);
            randfn=parameters['randList'];
            try:
                dat=np.loadtxt(randfn);
                try:
                    parameters['randList']=dat;
                except:
                    print('Cannot read random list',dat);
                    parameters['error']=1;
                    parameters=comm.bcast(parameters,root=0);
                    sys.exit(1);
            except:
                print('Cannot find random list file:',randfn);
                parameters['error']=1;
                parameters=comm.bcast(parameters,root=0);                
                sys.exit(1);
        else:
            if (parameters['gVar']!=0):
                randList=np.random.normal(1,parameters['gVar'],int(parameters['wireLength']));
                while not (np.prod(randList>0)):
                    randList=np.random.normal(1,parameters['gVar'],int(parameters['wireLength']));  
                parameters['randList']=randList;
                
            if (parameters['deltaVar']!=0):
                randList=np.random.normal(parameters['delta0'],parameters['deltaVar'],int(parameters['wireLength']));
                while not (np.prod(randList>0)):
                    randList=np.random.normal(parameters['delta0'],parameters['deltaVar'],int(parameters['wireLength']));
                parameters['randList']=randList;     
			

        print(parameters);   

        
    parameters=comm.bcast(parameters,root=0);
    if parameters['error']!=0:   #for the slave to exit
#        print('I am rank=',rank,'My flag is',parameters['error'],'I exit because',parameters['error']!=0);
        sys.exit(1);
    if parameters['muNum']==0:
        vz0=parameters['vz0'];
        tot=int(parameters['vzNum']);
        vzStep = parameters['vzStep'];  
    else:
        mu0=parameters['mu'];
        tot=int(parameters['muNum']); 
        muStep=parameters['muStep'];      
    np.warnings.filterwarnings('ignore');
    vBiasMin = parameters['vBiasMin']; 
    vBiasMax = parameters['vBiasMax']; 
    vBiasNumber = int(parameters['vBiasNum']);
    vBiasRange = np.linspace(vBiasMin, vBiasMax, vBiasNumber);    
    randList=parameters['randList'];    
    per=int(tot/size);    
    
    if parameters['leadNum']==1:
        leadPos=int(parameters['leadPos']);
        for irun in range(leadPos+1):
            parameters['leadPos']=irun;
            sendbuf=np.empty((per,vBiasNumber));  #conductance
            if parameters['isTV']==1:
                sendbuf2=np.empty((per,vBiasNumber)); #topological visibility
            
            for ii in range(per):
                if parameters['muNum']==0:
                    parameters['vz'] = vz0+(ii+rank*per)*vzStep;
                else:
                    parameters['mu'] = mu0+(ii+rank*per)*muStep;
                    
                if parameters['gVar']!=0:
                    parameters['randList']=randList*parameters['vz'];
                if parameters['isSE']==0:
                    junction=Maj.make_NS_junction(parameters);   #Change this if junction is voltage dependent, e.g. in Self energy
                for index in range(vBiasNumber):
                    vBias=vBiasRange[index];
                    parameters['vBias']=vBias;
                    if parameters['isSE']==1:
                        junction=Maj.make_NS_junction(parameters);                    
                    if parameters['isTV']==0:
                        sendbuf[ii,index]=Maj.conductance(parameters,junction);
                    else:
                        sendbuf[ii,index],sendbuf2[ii,index]=Maj.conductanceAndTV(parameters,junction);
            if (rank==0):
                recvbuf=np.empty((tot,vBiasNumber)); 
                if parameters['isTV']==1:
                    recvbuf2=np.empty((tot,vBiasNumber));
            else:
                recvbuf=None;
                if parameters['isTV']==1:
                    recvbuf2=None;
            comm.Gather(sendbuf,recvbuf,root=0);
            if parameters['isTV']==1:
                comm.Gather(sendbuf2,recvbuf2,root=0);
        
            if (rank==0):
                fn_mu=('m'+str(parameters['mu']))*(parameters['muNum']==0);
                fn_Delta='D'+str(parameters['delta0']);
                fn_alpha='a'+str(parameters['alpha_R']);
                fn_wl='L'+str(int(parameters['wireLength']));
                fn_potType=str(parameters['potType'])*(parameters['potType']!=0);
                fn_leadPos='L'*(parameters['leadPos']==0)+'R'*(parameters['leadPos']==1);
                if parameters['muNum']==0:
                    fn_range=('-'+str(vzStep*tot)+','+str(vBiasMax)+'-')*(parameters['muNum']==0);
                else:
                    fn_range=('-'+str(mu0)+','+str(mu0+muStep*tot)+','+str(vBiasMax)+'-')*(parameters['muNum']!=0);
                fn_muMax=('mx'+str(parameters['muMax']))*(parameters['potType']!=0);
                fn_potPeakPos=('pk'+str(parameters['potPeakPos']))*((parameters['potType']=='lorentz')+( parameters['potType']=='lorentzsigmoid'));
                fn_potSigma=('sg'+str(parameters['potSigma']))*((parameters['potType']=='exp')+(parameters['potType']=='sigmoid'));
                fn_muVar=('mVar'+str(parameters['muVar']))*(parameters['muVar']!=0)
                fn_dissipation=('G'+str(parameters['dissipation']))*(parameters['isDissipationVar']!=0);
                fn_dotLength=('dL'+str(int(parameters['dotLength'])))*(parameters['isQD']!=0);
                fn_qdPeak=('VD'+str(parameters['qdPeak']))*(parameters['isQD']!=0);
                fn_couplingSCSM=('g'+str(parameters['couplingSCSM']))*(parameters['isSE']==1);
                fn_vc=('vc'+str(parameters['vc']))*(parameters['isSE']==1)*(parameters['vc']!=0);
                fn_gVar=('gVar'+str(parameters['gVar']))*(parameters['gVar']!=0);
                fn_deltaVar=('DVar'+str(parameters['deltaVar']))*(parameters['deltaVar']!=0);
                
                fn=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_muMax+fn_potPeakPos+fn_potSigma+fn_muVar+fn_qdPeak+fn_dotLength+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+fn_leadPos+fn_range;
                
                np.savetxt(fn+'.dat',recvbuf);
                if parameters['isTV']==1:
                    np.savetxt(fn+'TV.dat',recvbuf2);
                if parameters['muNum']==0:
                    xRange=np.arange(tot)*vzStep;
                else:
                    xRange=mu0+np.arange(tot)*muStep;
                fig=plt.figure();
                plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbuf), cmap='rainbow');
                if parameters['muNum']==0:            
                    plt.xlabel('Vz(meV)');
                else:
                    plt.xlabel('mu(meV)');
                plt.ylabel('V_bias(meV)');
                plt.colorbar();
                plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax));
                fig.savefig(fn+'.png');
                
                if parameters['isTV']==1:
                    fig2=plt.figure();
                    plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbuf2));
                    plt.xlabel('Vz(meV)');
                    plt.ylabel('V_bias(meV)');
                    plt.colorbar();
                    plt.axis((0,tot*vzStep,vBiasMin,vBiasMax));
                    fig2.savefig(fn+'TV.png');
        
    elif parameters['leadNum']==2:
        sendbufGLL=np.empty((per,vBiasNumber));  #conductance
        sendbufGRR=np.empty((per,vBiasNumber));
        sendbufGLR=np.empty((per,vBiasNumber));
        sendbufGRL=np.empty((per,vBiasNumber));
        for ii in range(per):
            if parameters['muNum']==0:
                parameters['vz'] = vz0+(ii+rank*per)*vzStep;
            else:
                parameters['mu'] = mu0+(ii+rank*per)*muStep;
                
            if parameters['gVar']!=0:
                parameters['randList']=randList*parameters['vz'];
            if parameters['isSE']==0:
                junction=Maj.make_NS_junction(parameters);   #Change this if junction is voltage dependent, e.g. in Self energy
            for index in range(vBiasNumber):
                vBias=vBiasRange[index];
                parameters['vBias']=vBias;
                if parameters['isSE']==1:
                    junction=Maj.make_NS_junction(parameters);                    
                (sendbufGLL[ii,index],sendbufGRR[ii,index],sendbufGLR[ii,index],sendbufGRL[ii,index])=Maj.conductance_matrix(parameters,junction);
                    
            if (rank==0):
                recvbufGLL=np.empty((tot,vBiasNumber)); 
                recvbufGRR=np.empty((tot,vBiasNumber));
                recvbufGLR=np.empty((tot,vBiasNumber));
                recvbufGRL=np.empty((tot,vBiasNumber));
            else:
                recvbufGLL=None;
                recvbufGRR=None;
                recvbufGLR=None;
                recvbufGRL=None;

            comm.Gather(sendbufGLL,recvbufGLL,root=0);
            comm.Gather(sendbufGRR,recvbufGRR,root=0);
            comm.Gather(sendbufGLR,recvbufGLR,root=0);
            comm.Gather(sendbufGRL,recvbufGRL,root=0);

        
        if (rank==0):
            fn_mu=('m'+str(parameters['mu']))*(parameters['muNum']==0);
            fn_Delta='D'+str(parameters['delta0']);
            fn_alpha='a'+str(parameters['alpha_R']);
            fn_wl='L'+str(int(parameters['wireLength']));
            fn_potType=str(parameters['potType'])*(parameters['potType']!=0);
            if parameters['muNum']==0:
                fn_range=('-'+str(vzStep*tot)+','+str(vBiasMax)+'-')*(parameters['muNum']==0);
            else:
                fn_range=('-'+str(mu0)+','+str(mu0+muStep*tot)+','+str(vBiasMax)+'-')*(parameters['muNum']!=0);
            fn_muMax=('mx'+str(parameters['muMax']))*(parameters['potType']!=0);
            fn_potPeakPos=('pk'+str(parameters['potPeakPos']))*((parameters['potType']=='lorentz')+( parameters['potType']=='lorentzsigmoid'));
            fn_potSigma=('sg'+str(parameters['potSigma']))*((parameters['potType']=='exp')+(parameters['potType']=='sigmoid'));
            fn_muVar=('mVar'+str(parameters['muVar']))*(parameters['muVar']!=0)
            fn_dissipation=('G'+str(parameters['dissipation']))*(parameters['isDissipationVar']!=0);
            fn_dotLength=('dL'+str(int(parameters['dotLength'])))*(parameters['isQD']!=0);
            fn_qdPeak=('VD'+str(parameters['qdPeak']))*(parameters['isQD']!=0);
            fn_couplingSCSM=('g'+str(parameters['couplingSCSM']))*(parameters['isSE']==1);
            fn_vc=('vc'+str(parameters['vc']))*(parameters['isSE']==1)*(parameters['vc']!=0);
            fn_gVar=('gVar'+str(parameters['gVar']))*(parameters['gVar']!=0);
            fn_deltaVar=('DVar'+str(parameters['deltaVar']))*(parameters['deltaVar']!=0);
            
            fnLL=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_muMax+fn_potPeakPos+fn_potSigma+fn_muVar+fn_qdPeak+fn_dotLength+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+'LL'+fn_range;
            fnRR=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_muMax+fn_potPeakPos+fn_potSigma+fn_muVar+fn_qdPeak+fn_dotLength+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+'RR'+fn_range;
            fnLR=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_muMax+fn_potPeakPos+fn_potSigma+fn_muVar+fn_qdPeak+fn_dotLength+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+'LR'+fn_range;
            fnRL=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_muMax+fn_potPeakPos+fn_potSigma+fn_muVar+fn_qdPeak+fn_dotLength+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+'RL'+fn_range;
            
            np.savetxt(fnLL+'.dat',recvbufGLL);
            np.savetxt(fnRR+'.dat',recvbufGRR);
            np.savetxt(fnLR+'.dat',recvbufGLR);
            np.savetxt(fnRL+'.dat',recvbufGRL);

            if parameters['muNum']==0:
                xRange=np.arange(tot)*vzStep;
            else:
                xRange=mu0+np.arange(tot)*muStep;
            figLL=plt.figure();
            plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbufGLL), cmap='rainbow');
            if parameters['muNum']==0:            
                plt.xlabel('Vz(meV)');
            else:
                plt.xlabel('mu(meV)');
            plt.ylabel('V_bias(meV)');
            plt.colorbar();
            plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax));
            figLL.savefig(fnLL+'.png');
            
            figRR=plt.figure();
            plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbufGRR), cmap='rainbow');
            if parameters['muNum']==0:            
                plt.xlabel('Vz(meV)');
            else:
                plt.xlabel('mu(meV)');
            plt.ylabel('V_bias(meV)');
            plt.colorbar();
            plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax));
            figRR.savefig(fnRR+'.png');
            
            figLR=plt.figure();
            plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbufGLR), cmap='rainbow');
            if parameters['muNum']==0:            
                plt.xlabel('Vz(meV)');
            else:
                plt.xlabel('mu(meV)');
            plt.ylabel('V_bias(meV)');
            plt.colorbar();
            plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax));
            figLR.savefig(fnLR+'.png');
            
            figRL=plt.figure();
            plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbufGRL), cmap='rainbow');
            if parameters['muNum']==0:            
                plt.xlabel('Vz(meV)');
            else:
                plt.xlabel('mu(meV)');
            plt.ylabel('V_bias(meV)');
            plt.colorbar();
            plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax));
            figRL.savefig(fnRL+'.png');        
    
if __name__=="__main__":
	main()

