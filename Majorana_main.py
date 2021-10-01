import matplotlib
from pytest import param
matplotlib.use('Agg')
from mpi4py import MPI
import numpy as np
import Majorana_module as Maj
import sys
import re
import matplotlib.pyplot as plt
import time
import argparse
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size=comm.Get_size()

def main():
    vars=len(sys.argv)
    parameters = {'mass':0.01519,'a':1,'mu':.2,'alpha_R':5, 'delta0':0.2,'wireLength':1000,
               'muLead':25.0, 'barrierNum':2,'barrierE':10.0, 'dissipation':0.0001,'isDissipationVar':0,'barrierRelative':0,
               'isQD':0, 'qdPeak':0.4, 'qdLength':20, 'qdPeakR':0,'qdLengthR':0,
               'isSE':0, 'couplingSCSM':0.2, 'vc':0,
               'potType':0,'potPeakPos':0,'potSigma':1,'potPeak':0,'potPeakR':0,'potPeakPosR':0,'potSigmaR':0,
               'muVar':0,'muVarList':0,'muVarType':0,'scatterList':0,'N_muVar':1,
               'gVar':0,'randList':0,
               'deltaVar':0,
               'couplingSCSMVar':0,
               'vz':0.0, 'vBias':0.0,
               'leadPos':0,'leadNum':1,
               'isS':0,
               'x':'vz','xMin':0,'xMax':2.048,'xNum':256,'xUnit':'meV',
               'y':'vBias','yMin':-.3,'yMax':.3,'yNum':301,'yUnit':'mV',
               'alpha':-1,
               'colortheme':'seismic','vmin':0,'vmax':4,
               'error':0}
    if (rank==0):
        if vars>1:
            #read and parse parameters
            for i in range(1,vars):
                try:
                    varName=re.search('(.)*(?=\=)',sys.argv[i]).group(0)
                    varValue=re.search('(?<=\=)(.)*',sys.argv[i]).group(0)
                    if varName in parameters:
                        if varName in ['potType','muVarList','randList','muVarType','scatterList','xUnit','yUnit','x','y']:
                            parameters[varName]=varValue
                        else:
                            parameters[varName]=float(varValue)
                    else:
                        print('Cannot find the parameter(s)',varName)
                        parameters['error']=1
                        parameters=comm.bcast(parameters,root=0)
                        sys.exit(1)
                except:
                    print('Cannot parse the input parameters',sys.argv[i])
                    parameters['error']=1
                    parameters=comm.bcast(parameters,root=0)
                    sys.exit(1)

        if (parameters['muVarType']==0 or parameters['muVarType']=='Gaussian'):
            if (isinstance(parameters['muVarList'],str)):                   #whether use disorder configuration file
                print('gaussian disorder in the chemical potential uses filename: '+parameters['muVarList'])
                muVarfn=parameters['muVarList']
                try:
                    dat=np.loadtxt(muVarfn)
                    try:
                        parameters['muVarList']=dat
                    except:
                        print('Cannot read muVarList: ',dat)
                        parameters['error']=1
                        parameters=comm.bcast(parameters,root=0)
                        sys.exit(1)
                except:
                    print('Cannot find disorder file: ',muVarfn)
                    parameters['error']=1
                    parameters=comm.bcast(parameters,root=0)
                    sys.exit(1)
            else:
                if (parameters['muVar']!=0):
                    parameters['muVarList']=np.random.normal(0,parameters['muVar'],int(parameters['wireLength']))

        if (parameters['muVarType']=='Coulomb'):
            if (isinstance(parameters['scatterList'],str)):
                print('Coulomb disorder in the chemical potential uses filename: '+parameters['scatterList'])
                scatterListfn=parameters['scatterList']            #the position of scattering center in (um)
                try:
                    dat=np.loadtxt(scatterListfn)
                    parameters['scatterList']=dat
                    try:
                        prefactor=1/(4*np.pi*8.85418781762039e-12)*(1.60217662e-19/(5*parameters['a']*10e-9))*1e3       #1/(4pi*epsilon)*e/(5*a) (meV)
                        parameters['muVarList']=prefactor*np.array([np.sum(1/np.array([np.sqrt((site-xi/(parameters['a']*1e-2))**2+(5)**2) for xi in dat])) for site in range(int(parameters['wireLength']))])
                    except:
                        print('Cannot read scatterList: ',dat)
                        parameters['error']=1
                        parameters=comm.bcast(parameters,root=0)
                        sys.exit(1)
                except:
                    print('Cannot find disorder file: ',scatterListfn)
                    parameters['error']=1
                    parameters=comm.bcast(parameters,root=0)
                    sys.exit(1)
            else:
                if (parameters['muVar']!=0):
                    dat=np.random.uniform(low=0,high=parameters['wireLength'],size=int(parameters['muVar']))*parameters['a']*1e-2
                    parameters['scatterList']=dat
                    prefactor=1/(4*np.pi*8.85418781762039e-12)*(1.60217662e-19/(5*parameters['a']*10e-9))*1e3
                    parameters['muVarList']=prefactor*np.array([np.sum(1/np.array([np.sqrt((site-xi/(parameters['a']*1e-2))**2+(5)**2) for xi in dat])) for site in range(int(parameters['wireLength']))])

        if (isinstance(parameters['randList'],str)):
            print('random list use filename:'+parameters['randList'])
            randfn=parameters['randList']
            try:
                dat=np.loadtxt(randfn)
                try:
                    parameters['randList']=dat
                except:
                    print('Cannot read random list',dat)
                    parameters['error']=1
                    parameters=comm.bcast(parameters,root=0)
                    sys.exit(1)
            except:
                print('Cannot find random list file:',randfn)
                parameters['error']=1
                parameters=comm.bcast(parameters,root=0)
                sys.exit(1)
        else:
            if (parameters['gVar']!=0):
                randList=np.random.normal(1,parameters['gVar'],int(parameters['wireLength']))
                while not (np.prod(randList>0)):
                    randList=np.random.normal(1,parameters['gVar'],int(parameters['wireLength']))
                parameters['randList']=randList

            if (parameters['deltaVar']!=0):
                randList=np.random.normal(parameters['delta0'],parameters['deltaVar'],int(parameters['wireLength']))
                while not (np.prod(randList>0)):
                    randList=np.random.normal(parameters['delta0'],parameters['deltaVar'],int(parameters['wireLength']))
                parameters['randList']=randList

            if (parameters['couplingSCSMVar']!=0):
                randList=np.random.normal(parameters['couplingSCSM'],parameters['couplingSCSMVar'],int(parameters['wireLength']))
                while not (np.prod(randList>0)):
                    randList=np.random.normal(parameters['couplingSCSM'],parameters['couplingSCSMVar'],int(parameters['wireLength']))
                parameters['randList']=randList

        if parameters['x'] not in ['mu','alpha_R','delta0',\
        'muLead','barrierE','dissipation',\
        'qdPeak','qdLength','qdPeakR','qdLengthR',\
        'couplingSCSM','vc',\
        'potPeakPos','potSigma','potPeak','potPeakR','potPeakPosR','potSigmaR',
        'vz']:
            print('The x axis does not support the knob: '+parameters['x'])
            parameters['error']=1
            parameters=comm.bcast(parameters,root=0)
            sys.exit(1)

        print(parameters)

    parameters=comm.bcast(parameters,root=0)
    if parameters['error']!=0:   #for the slave to exit
#        print('I am rank=',rank,'My flag is',parameters['error'],'I exit because',parameters['error']!=0)
        sys.exit(1)

    tot=int(parameters['xNum'])
    xStep=(parameters['xMax']-parameters['xMin'])/parameters['xNum']

    # np.warnings.filterwarnings('ignore')
    parameters['yNum']=int(parameters['yNum'])
    yRange=np.linspace(parameters['yMin'],parameters['yMax'],parameters['yNum'])

    randList=parameters['randList']
    
    if tot%size>0:
        raise ValueError('The total number on x axis ({:d}) is not multiple of the cores requested ({:d})'.format(parameters['xNum'],size))
    else:
        per=tot//size

    if parameters['leadNum']==1:
        leadPos=int(parameters['leadPos'])
        for irun in range(leadPos+1):
            parameters['leadPos']=irun
            sendbuf=np.empty((per,parameters['yNum']))  #conductance
            

            for ii in range(per):
                parameters[parameters['x']]=parameters['xMin']+(ii+rank*per)*xStep
                if parameters['gVar']!=0:
                    parameters['randList']=randList*parameters['vz']
                if parameters['isSE']==0 and parameters['y']=='vBias':
                    junction=Maj.make_NS_junction(parameters)   #Change this if junction is voltage dependent, e.g. in Self energy
                for index in range(parameters['yNum']):
                    parameters[parameters['y']]=yRange[index]
                    if parameters['barrierRelative']!=0:
                        parameters['barrierE']=parameters['mu']+parameters['barrierRelative']
                    if not (parameters['isSE']==0 and parameters['y']=='vBias'):
                        junction=Maj.make_NS_junction(parameters)

                    sendbuf[ii,index]=Maj.conductance(parameters,junction)

                    assert parameters['isS']!=0,'The number of lead should be 2 when export Smatrix'

            if (rank==0):
                recvbuf=np.empty((tot,parameters['yNum']))
            else:
                recvbuf=None
            comm.Gather(sendbuf,recvbuf,root=0)

            if (rank==0):
                fn={}
                fn['mu']=('m'+str(parameters['mu']))
                fn['delta0']='D'+str(parameters['delta0'])
                fn['alpha_R']='a'+str(parameters['alpha_R'])
                fn['wireLength']='L'+str(int(parameters['wireLength']))
                fn['muLead']='muL'+str(parameters['muLead'])
                fn['barrierE']=('bE'+str(parameters['barrierE']))*(parameters['barrierRelative']==0)
                fn['barrierRelative']=('bR'+str(parameters['barrierRelative']))*(parameters['barrierRelative']!=0)
                fn['potType']=str(parameters['potType'])*(parameters['potType']!=0)
                fn['leadPos']='L'*(parameters['leadPos']==0)+'R'*(parameters['leadPos']==1)
                fn['range']=('-'+parameters['x']+'('+str(parameters['xMin'])+','+str(parameters['xMax'])+')'+','+parameters['y']+'('+str(parameters['yMin'])+','+str(parameters['yMax'])+')'+'-')
                fn['potPeak']=('mx'+str(parameters['potPeak']))*(parameters['potType']!=0)
                fn['potPeakPos']=('pk'+str(parameters['potPeakPos']))*((parameters['potType']=='lorentz')+( parameters['potType']=='lorentzsigmoid')+(parameters['potType']=='exp2'))
                fn['potSigma']=('sg'+str(parameters['potSigma']))*((parameters['potType']=='exp')+(parameters['potType']=='sigmoid')+(parameters['potType']=='exp2')+(parameters['potType']=='cos')+(parameters['potType']=='cos2'))
                fn['potPeakR']=('mxR'+str(parameters['potPeakR']))*(parameters['potType']=='exp2')
                fn['potPeakPosR']=('pkR'+str(parameters['potPeakPosR']))*( parameters['potType']=='exp2')
                fn['potSigmaR']=('sgR'+str(parameters['potSigmaR']))*(parameters['potType']=='exp2')
                fn['muVar']=('mVar'+str(parameters['muVar']))*(parameters['muVar']!=0)
                fn['muVarType']=('C')*(parameters['muVarType']=='Coulomb')
                fn['dissipation']=('G'+str(parameters['dissipation']))
                fn['qdLength']=('dL'+str(int(parameters['qdLength'])))*(parameters['isQD']!=0)
                fn['qdPeak']=('VD'+str(parameters['qdPeak']))*(parameters['isQD']!=0)
                fn['qdLengthR']=('dLR'+str(int(parameters['qdLengthR'])))*(parameters['isQD']!=0)*(parameters['qdLengthR']!=0)
                fn['qdPeakR']=('VDR'+str(parameters['qdPeakR']))*(parameters['isQD']!=0)*(parameters['qdLengthR']!=0)
                fn['couplingSCSM']=('g'+str(parameters['couplingSCSM']))*(parameters['isSE']==1)
                fn['vc']=('vc'+str(parameters['vc']))*(parameters['isSE']==1)*(parameters['vc']!=0)
                fn['gVar']=('gVar'+str(parameters['gVar']))*(parameters['gVar']!=0)
                fn['deltaVar']=('DVar'+str(parameters['deltaVar']))*(parameters['deltaVar']!=0)
                fn['couplingSCSMVar']=('gammaVar'+str(parameters['couplingSCSMVar']))*(parameters['couplingSCSMVar']!=0)
                fn['vz']=('vz'+str(parameters['vz']))
                fn['alpha']=('alpha'+str(parameters['alpha']))*(parameters['alpha']<=1 and parameters['alpha']>=0)
                fn[parameters['x']]=''
                fn[parameters['y']]=''                
                fn=fn['vz']+fn['mu']+fn['delta0']+fn['deltaVar']+fn['alpha_R']+fn['wireLength']+fn['muLead']+fn['potType']+fn['potPeak']+fn['potPeakPos']+fn['potSigma']+fn['potPeakR']+fn['potPeakPosR']+fn['potSigmaR']+fn['muVarType']+fn['muVar']+fn['alpha']+fn['qdPeak']+fn['qdLength']+fn['qdPeakR']+fn['qdLengthR']+fn['couplingSCSM']+fn['couplingSCSMVar']+fn['vc']+fn['dissipation']+fn['gVar']+fn['barrierE']+fn['range']+fn['leadPos']

                xRange=np.linspace(parameters['xMin'],parameters['xMax'],tot)
                fig,ax=plt.subplots()
                im=ax.pcolormesh(xRange,yRange,np.transpose(recvbuf), cmap=parameters['colortheme'],vmin=parameters['vmin'],vmax=parameters['vmax'],shading='auto')
                ax.set_xlabel('{}({})'.format(parameters['x'],parameters['xUnit']))
                ax.set_ylabel('{}({})'.format(parameters['y'],parameters['yUnit']))
                axins=ax.inset_axes([1.02,0,.05,1],transform=ax.transAxes)
                cb=plt.colorbar(im,cax=axins,ticks=[0,2,4])
                cb.ax.set_title(r'$G(e^2/h)$')                
                fig.savefig(fn+'.png',bbox_inches='tight')

    elif parameters['leadNum']==2:
        sendbufGLL=np.empty((per,parameters['yNum']))  #conductance
        sendbufGRR=np.empty((per,parameters['yNum']))
        sendbufGLR=np.empty((per,parameters['yNum']))
        sendbufGRL=np.empty((per,parameters['yNum']))
        if (parameters['isS']!=0):
            sendbufS=np.empty((per,8,8),dtype='complex128')
            sendbufTVL=np.empty(per)
            sendbufTVR=np.empty(per)
        for ii in range(per):
            parameters[parameters['x']]=parameters['xMin']+(ii+rank*per)*xStep
            if parameters['gVar']!=0:
                parameters['randList']=randList*parameters['vz']
            if parameters['isSE']==0 and parameters['y']=='vBias':
                junction=Maj.make_NS_junction(parameters)   #Change this if junction is voltage dependent, e.g. in Self energy
            for index in range(parameters['yNum']):
                parameters[parameters['y']]=yRange[index]
                if parameters['barrierRelative']!=0:
                    parameters['barrierE']=parameters['mu']+parameters['barrierRelative']                
                if not (parameters['isSE']==0 and parameters['y']=='vBias'):
                    junction=Maj.make_NS_junction(parameters)
                (sendbufGLL[ii,index],sendbufGRR[ii,index],sendbufGLR[ii,index],sendbufGRL[ii,index])=Maj.conductance_matrix(parameters,junction)
                if (parameters['isS']!=0):
                    if (parameters['vBias']==0):
                        sendbufS[ii,:,:],sendbufTVL[ii],sendbufTVR[ii]=Maj.getSMatrix(parameters, junction)


            if (rank==0):
                recvbufGLL=np.empty((tot,parameters['yNum']))
                recvbufGRR=np.empty((tot,parameters['yNum']))
                recvbufGLR=np.empty((tot,parameters['yNum']))
                recvbufGRL=np.empty((tot,parameters['yNum']))
                if (parameters['isS']!=0):
                    recvbufS=np.empty((tot,8,8),dtype='complex128')
                    recvbufTVL=np.empty(tot)
                    recvbufTVR=np.empty(tot)
            else:
                recvbufGLL=None
                recvbufGRR=None
                recvbufGLR=None
                recvbufGRL=None
                if (parameters['isS']!=0):
                    recvbufS=None
                    recvbufTVL=None
                    recvbufTVR=None

            comm.Gather(sendbufGLL,recvbufGLL,root=0)
            comm.Gather(sendbufGRR,recvbufGRR,root=0)
            comm.Gather(sendbufGLR,recvbufGLR,root=0)
            comm.Gather(sendbufGRL,recvbufGRL,root=0)
            if (parameters['isS']!=0):
                comm.Gather(sendbufS,recvbufS,root=0)
                comm.Gather(sendbufTVL,recvbufTVL,root=0)
                comm.Gather(sendbufTVR,recvbufTVR,root=0)

        if (rank==0):
            fn={}
            fn['mu']=('m'+str(parameters['mu']))
            fn['delta0']='D'+str(parameters['delta0'])
            fn['alpha_R']='a'+str(parameters['alpha_R'])
            fn['wireLength']='L'+str(int(parameters['wireLength']))
            fn['muLead']='muL'+str(parameters['muLead'])
            fn['barrierE']=('bE'+str(parameters['barrierE']))*(parameters['barrierRelative']==0)
            fn['barrierRelative']=('bR'+str(parameters['barrierRelative']))*(parameters['barrierRelative']!=0)
            fn['potType']=str(parameters['potType'])*(parameters['potType']!=0)
            fn['range']=('-'+parameters['x']+'('+str(parameters['xMin'])+','+str(parameters['xMax'])+')'+','+parameters['y']+'('+str(parameters['yMin'])+','+str(parameters['yMax'])+')'+'-')
            fn['potPeak']=('mx'+str(parameters['potPeak']))*(parameters['potType']!=0)
            fn['potPeakPos']=('pk'+str(parameters['potPeakPos']))*((parameters['potType']=='lorentz')+( parameters['potType']=='lorentzsigmoid')+(parameters['potType']=='exp2'))
            fn['potSigma']=('sg'+str(parameters['potSigma']))*((parameters['potType']=='exp')+(parameters['potType']=='sigmoid')+(parameters['potType']=='exp2')+(parameters['potType']=='cos')+(parameters['potType']=='cos2'))
            fn['potPeakR']=('mxR'+str(parameters['potPeakR']))*(parameters['potType']=='exp2')
            fn['potPeakPosR']=('pkR'+str(parameters['potPeakPosR']))*( parameters['potType']=='exp2')
            fn['potSigmaR']=('sgR'+str(parameters['potSigmaR']))*(parameters['potType']=='exp2')
            fn['muVar']=('mVar'+str(parameters['muVar']))*(parameters['muVar']!=0)
            fn['muVarType']=('C')*(parameters['muVarType']=='Coulomb')
            fn['dissipation']=('G'+str(parameters['dissipation']))
            fn['qdLength']=('dL'+str(int(parameters['qdLength'])))*(parameters['isQD']!=0)
            fn['qdPeak']=('VD'+str(parameters['qdPeak']))*(parameters['isQD']!=0)
            fn['qdLengthR']=('dLR'+str(int(parameters['qdLengthR'])))*(parameters['isQD']!=0)*(parameters['qdLengthR']!=0)
            fn['qdPeakR']=('VDR'+str(parameters['qdPeakR']))*(parameters['isQD']!=0)*(parameters['qdLengthR']!=0)
            fn['couplingSCSM']=('g'+str(parameters['couplingSCSM']))*(parameters['isSE']==1)
            fn['vc']=('vc'+str(parameters['vc']))*(parameters['isSE']==1)*(parameters['vc']!=0)
            fn['gVar']=('gVar'+str(parameters['gVar']))*(parameters['gVar']!=0)
            fn['deltaVar']=('DVar'+str(parameters['deltaVar']))*(parameters['deltaVar']!=0)
            fn['couplingSCSMVar']=('gammaVar'+str(parameters['couplingSCSMVar']))*(parameters['couplingSCSMVar']!=0)
            fn['vz']=('vz'+str(parameters['vz']))
            fn['alpha']=('alpha'+str(parameters['alpha']))*(parameters['alpha']<=1 and parameters['alpha']>=0)
            
            fn[parameters['x']]=''
            fn[parameters['y']]=''
            
            fn=fn['vz']+fn['mu']+fn['delta0']+fn['deltaVar']+fn['alpha_R']+fn['wireLength']+fn['muLead']+fn['potType']+fn['potPeak']+fn['potPeakPos']+fn['potSigma']+fn['potPeakR']+fn['potPeakPosR']+fn['potSigmaR']+fn['muVarType']+fn['muVar']+fn['alpha']+fn['qdPeak']+fn['qdLength']+fn['qdPeakR']+fn['qdLengthR']+fn['couplingSCSM']+fn['couplingSCSMVar']+fn['vc']+fn['dissipation']+fn['gVar']+fn['barrierE']+fn['range']
            fnLL=fn+'LL'
            fnRR=fn+'RR'
            fnLR=fn+'LR'
            fnRL=fn+'RL'

            np.savetxt(fnLL+'.dat',recvbufGLL)
            np.savetxt(fnRR+'.dat',recvbufGRR)
            np.savetxt(fnLR+'.dat',recvbufGLR)
            np.savetxt(fnRL+'.dat',recvbufGRL)
            if (parameters['isS']!=0):
                with open(fn+'s.pickle','wb') as f:
                    pickle.dump([recvbufS,recvbufTVL,recvbufTVR],f)

            xRange=np.linspace(parameters['xMin'],parameters['xMax'],tot)
            fig,ax=plt.subplots(2,2,sharex=True,sharey=True,tight_layout=True)
            im=[ax.pcolormesh(xRange,yRange,data.T,cmap=parameters['colortheme'],vmin=parameters['vmin'],vmax=parameters['vmax'],shading='auto') for ax,data in zip(ax[0,:],(recvbufGLL,recvbufGRR))]
            im.append(ax[1,0].pcolormesh(xRange,yRange,recvbufGLR.T,cmap=parameters['colortheme'],shading='auto'))
            im.append(ax[1,1].pcolormesh(xRange,yRange,recvbufGRL.T,cmap=parameters['colortheme'],shading='auto'))
            [ax.set_xlabel('{}({})'.format(parameters['x'],parameters['xUnit'])) for ax in ax[1,:]]
            [ax.set_ylabel('{}({})'.format(parameters['y'],parameters['yUnit'])) for ax in ax[:,0]]
            axins=[ax.inset_axes([1.02,0,.05,1],transform=ax.transAxes) for ax in ax.flatten()]
            cb=[plt.colorbar(im,cax=axins) for im,axins in zip(im,axins)]
            [cb.ax.set_title(r'$G(e^2/h)$') for cb in cb]
            [ax.text(.5,1,text,transform=ax.transAxes,va='bottom',ha='center') for ax,text in zip(ax.flatten(),('LL','RR','LR','RL'))]
            fig.savefig(fn+'.png',bbox_inches='tight')

            if (parameters['isS']==1):
                figTV,ax=plt.subplots()
                ax.plot(xRange,recvbufTVL,lw=1,color='r',label='L')
                ax.plot(xRange,recvbufTVR,lw=1,color='b',label='R')
                ax.set_xlabel('{}({})'.format(parameters['y'],parameters['yUnit']))
                ax.set_ylabel('TV')
                ax.legend()
                # plt.axis((xRange[0],xRange[-1],-1,1))
                figTV.savefig(fn+'TV.png')

if __name__=="__main__":
    main()
