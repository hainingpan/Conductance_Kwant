import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
import numpy as np
import Majorana_module as Maj
import sys
import re
import matplotlib.pyplot as plt
import time
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size=comm.Get_size()

def main():
    vars=len(sys.argv)
    parameters = {'isTV':0,'a':1,'mu':.2,'alpha_R':5, 'delta0':0.2,'wireLength':1000,
               'muLead':25.0, 'barrierNum':2,'barrierE':10.0, 'dissipation':0.0001,'isDissipationVar':0,
               'isQD':0, 'qdPeak':0.4, 'qdLength':20, 'qdPeakR':0,'qdLengthR':0,
               'isSE':0, 'couplingSCSM':0.2, 'vc':0,
               'potType':0,'potPeakPos':0,'potSigma':1,'potPeak':0,'potPeakR':0,'potPeakPosR':0,'potSigmaR':0,
               'muVar':0,'muVarList':0,'muVarType':0,'scatterList':0,'N_muVar':1,
               'gVar':0,'randList':0,
               'deltaVar':0,
               'vz':0.0, 'vBias':0.0,'vBiasMin':-0.3,'vBiasMax':0.3,'vBiasNum':1001,
               # 'vz0':0,'vzNum':256,'vzStep': 0.002,'mu0':0,'muMax':1,'muStep':0.002,'muNum':0,
               'leadPos':0,'leadNum':1,
               'Q':0,
               'x':'vz','y':0,'xMin':0,'xMax':2.048,'xNum':256,'xUnit':'meV',
               'error':0}
    if (rank==0):
        if vars>1:
            #read and parse parameters
            for i in range(1,vars):
                try:
                    varName=re.search('(.)*(?=\=)',sys.argv[i]).group(0)
                    varValue=re.search('(?<=\=)(.)*',sys.argv[i]).group(0)
                    if varName in parameters:
                        if varName in ['potType','muVarList','randList','muVarType','scatterList','xUnit','x']:
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

    # if parameters['muNum']==0:
    #     vz0=parameters['vz0']
    #     tot=int(parameters['vzNum'])
    #     vzStep = parameters['vzStep']
    # else:
    #     mu0=parameters['mu0']
    #     tot=int(parameters['muNum'])
    #     muStep=parameters['muStep']
    np.warnings.filterwarnings('ignore')
    vBiasMin = parameters['vBiasMin']
    vBiasMax = parameters['vBiasMax']
    vBiasNumber = int(parameters['vBiasNum'])
    vBiasRange = np.linspace(vBiasMin, vBiasMax, vBiasNumber)
    randList=parameters['randList']
    per=int(tot/size)

    if parameters['leadNum']==1:
        leadPos=int(parameters['leadPos'])
        for irun in range(leadPos+1):
            parameters['leadPos']=irun
            sendbuf=np.empty((per,vBiasNumber))  #conductance
            if (parameters['Q']!=0):
                sendbufQ=np.empty((per,1))
            if parameters['isTV']==1:
                sendbuf2=np.empty((per,1)) #topological visibility

            for ii in range(per):
                parameters[parameters['x']]=parameters['xMin']+(ii+rank*per)*xStep
                # if parameters['muNum']==0:
                #     parameters['vz'] = vz0+(ii+rank*per)*vzStep
                # else:
                #     parameters['mu'] = mu0+(ii+rank*per)*muStep

                if parameters['gVar']!=0:
                    parameters['randList']=randList*parameters['vz']
                if parameters['isSE']==0:
                    junction=Maj.make_NS_junction(parameters)   #Change this if junction is voltage dependent, e.g. in Self energy
                for index in range(vBiasNumber):
                    vBias=vBiasRange[index]
                    parameters['vBias']=vBias
                    if parameters['isSE']==1:
                        junction=Maj.make_NS_junction(parameters)

                    sendbuf[ii,index]=Maj.conductance(parameters,junction)
                    if parameters['isTV']!=0:
                        if (vBias==0):
                            sendbuf2[ii,:]=Maj.TV(parameters,junction)

                    if (parameters['Q']!=0):
                        if (vBias==0):
                            sendbufQ[ii,:]=Maj.topologicalQ(parameters,junction)

            if (rank==0):
                recvbuf=np.empty((tot,vBiasNumber))
                if (parameters['Q']!=0):
                    recvbufQ=np.empty((tot,1))
                if parameters['isTV']!=0:
                    recvbuf2=np.empty((tot,1))
            else:
                recvbuf=None
                if (parameters['Q']!=0):
                    recvbufQ=None
                if parameters['isTV']==1:
                    recvbuf2=None
            comm.Gather(sendbuf,recvbuf,root=0)
            if (parameters['Q']!=0):
                comm.Gather(sendbufQ,recvbufQ,root=0)
            if parameters['isTV']!=0:
                comm.Gather(sendbuf2,recvbuf2,root=0)

            if (rank==0):
                fn_mu=('m'+str(parameters['mu']))*(parameters['x']!='delta0')
                fn_Delta='D'+str(parameters['delta0'])*(parameters['x']!='delta0')
                fn_alpha='a'+str(parameters['alpha_R'])*(parameters['x']!='alpha_R')
                fn_wl='L'+str(int(parameters['wireLength']))
                fn_muLead='muL'+str(parameters['muLead'])*(parameters['x']!='muLead')
                fn_bE=('bE'+str(parameters['barrierE']))*(parameters['x']!='barrierE')
                fn_potType=str(parameters['potType'])*(parameters['potType']!=0)
                fn_leadPos='L'*(parameters['leadPos']==0)+'R'*(parameters['leadPos']==1)
                fn_range=('-'+parameters['x']+'('+str(parameters['xMin'])+','+str(parameters['xMax'])+')'+','+str(vBiasMax)+'-')
                # if parameters['muNum']==0:
                #     fn_range=('-'+str(vzStep*tot)+','+str(vBiasMax)+'-')*(parameters['muNum']==0)
                # else:
                #     fn_range=('-'+str(mu0)+','+str(mu0+muStep*tot)+','+str(vBiasMax)+'-')*(parameters['muNum']!=0)
                fn_potPeak=('mx'+str(parameters['potPeak']))*(parameters['potType']!=0)*(parameters['x']!='potPeak')
                fn_potPeakPos=('pk'+str(parameters['potPeakPos']))*((parameters['potType']=='lorentz')+( parameters['potType']=='lorentzsigmoid')+(parameters['potType']=='exp2'))*(parameters['x']!='potPeakPos')
                fn_potSigma=('sg'+str(parameters['potSigma']))*((parameters['potType']=='exp')+(parameters['potType']=='sigmoid')+(parameters['potType']=='exp2')+(parameters['potType']=='cos')+(parameters['potType']=='cos2'))*(parameters['x']!='potSigma')
                fn_potPeakR=('mxR'+str(parameters['potPeakR']))*(parameters['potType']=='exp2')*(parameters['x']!='potPeakR')
                fn_potPeakPosR=('pkR'+str(parameters['potPeakPosR']))*( parameters['potType']=='exp2')*(parameters['x']!='potPeakPosR')
                fn_potSigmaR=('sgR'+str(parameters['potSigmaR']))*(parameters['potType']=='exp2')*(parameters['x']!='potSigmaR')
                fn_muVar=('mVar'+str(parameters['muVar']))*(parameters['muVar']!=0)
                fn_muVarType=('C')*(parameters['muVarType']=='Coulomb')
                fn_dissipation=('G'+str(parameters['dissipation']))*(parameters['x']!='dissipation')
                fn_qdLength=('dL'+str(int(parameters['qdLength'])))*(parameters['isQD']!=0)*(parameters['x']!='qdLength')
                fn_qdPeak=('VD'+str(parameters['qdPeak']))*(parameters['isQD']!=0)*(parameters['x']!='qdPeak')
                fn_qdLengthR=('dLR'+str(int(parameters['qdLengthR'])))*(parameters['isQD']!=0)*(parameters['qdLengthR']!=0)*(parameters['x']!='qdLengthR')
                fn_qdPeakR=('VDR'+str(parameters['qdPeakR']))*(parameters['isQD']!=0)*(parameters['qdLengthR']!=0)*(parameters['x']!='qdPeakR')
                fn_couplingSCSM=('g'+str(parameters['couplingSCSM']))*(parameters['isSE']==1)*(parameters['x']!='couplingSCSM')
                fn_vc=('vc'+str(parameters['vc']))*(parameters['isSE']==1)*(parameters['vc']!=0)*(parameters['x']!='vc')
                fn_gVar=('gVar'+str(parameters['gVar']))*(parameters['gVar']!=0)
                fn_deltaVar=('DVar'+str(parameters['deltaVar']))*(parameters['deltaVar']!=0)
                fn_vz=('vz'+str(parameters['vz']))*(parameters['x']!='vz')
                fn=fn_vz+fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_muLead+fn_potType+fn_potPeak+fn_potPeakPos+fn_potSigma+fn_potPeakR+fn_potPeakPosR+fn_potSigmaR+fn_muVarType+fn_muVar+fn_qdPeak+fn_qdLength+fn_qdPeakR+fn_qdLengthR+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+fn_bE+fn_leadPos+fn_range

                np.savetxt(fn+'.dat',recvbuf)
                if (parameters['Q']!=0):
                   np.savetxt(fn+'Q.dat',recvbufQ)

                if parameters['isTV']==1:
                    np.savetxt(fn+'TV.dat',recvbuf2)

                # if parameters['muNum']==0:
                #     xRange=np.arange(tot)*vzStep
                # else:
                #     xRange=mu0+np.arange(tot)*muStep
                xRange=np.linspace(parameters['xMin'],parameters['xMax'],tot)
                fig=plt.figure()
                plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbuf), cmap='rainbow')
                # if parameters['muNum']==0:
                #     plt.xlabel('Vz(meV)')
                # else:
                #     plt.xlabel('mu(meV)')
                plt.xlabel(parameters['x']+'('+parameters['xUnit']+')')
                plt.ylabel(r'$V_\mathrm{bias}$ (meV)')
                plt.colorbar()
                plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax))
                fig.savefig(fn+'.png')

                if (parameters['Q']!=0):
                    figQ=plt.figure()
                    plt.plot(xRange,recvbufQ)
                    plt.xlabel('Vz(meV)')
                    plt.ylabel('det(r)')
                    plt.axis((xRange[0],xRange[-1],-1,1))
                    figQ.savefig(fn+'Q.png')

                if parameters['isTV']!=0:
                    fig2=plt.figure()
                    # plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbuf2))
                    plt.plot(xRange,recvbuf2)
                    plt.xlabel('Vz(meV)')
                    plt.ylabel(r'$V_\mathrm{bias}$ (meV)')
                    # plt.colorbar()
                    # plt.axis((0,tot*vzStep,vBiasMin,vBiasMax))
                    plt.axis((xRange[0],xRange[-1],-1,1))
                    fig2.savefig(fn+'TV.png')

    elif parameters['leadNum']==2:
        sendbufGLL=np.empty((per,vBiasNumber))  #conductance
        sendbufGRR=np.empty((per,vBiasNumber))
        sendbufGLR=np.empty((per,vBiasNumber))
        sendbufGRL=np.empty((per,vBiasNumber))
        if (parameters['Q']!=0):
            sendbufQ=np.empty((per,1))          #fix phase
            sendbufR=np.empty((per,8*8*2))
        for ii in range(per):
            parameters[parameters['x']]=parameters['xMin']+(ii+rank*per)*xStep
            # if parameters['muNum']==0:
            #     parameters['vz'] = vz0+(ii+rank*per)*vzStep
            # else:
            #     parameters['mu'] = mu0+(ii+rank*per)*muStep

            if parameters['gVar']!=0:
                parameters['randList']=randList*parameters['vz']
            if parameters['isSE']==0:
                junction=Maj.make_NS_junction(parameters)   #Change this if junction is voltage dependent, e.g. in Self energy
            for index in range(vBiasNumber):
                vBias=vBiasRange[index]
                parameters['vBias']=vBias
                if parameters['isSE']==1:
                    junction=Maj.make_NS_junction(parameters)
                (sendbufGLL[ii,index],sendbufGRR[ii,index],sendbufGLR[ii,index],sendbufGRL[ii,index])=Maj.conductance_matrix(parameters,junction)
                if (parameters['Q']!=0):
                    if (vBias==0):
                        # sendbufQ[ii,:]=Maj.topologicalQ(parameters,junction)
                        sendbufR[ii,:],sendbufQ[ii,:]=Maj.getSMatrix(parameters,junction)


            if (rank==0):
                recvbufGLL=np.empty((tot,vBiasNumber))
                recvbufGRR=np.empty((tot,vBiasNumber))
                recvbufGLR=np.empty((tot,vBiasNumber))
                recvbufGRL=np.empty((tot,vBiasNumber))
                if (parameters['Q']!=0):
                    recvbufQ=np.empty((tot,1))
                    recvbufR=np.empty((tot,8*8*2))
            else:
                recvbufGLL=None
                recvbufGRR=None
                recvbufGLR=None
                recvbufGRL=None
                if (parameters['Q']!=0):
                    recvbufQ=None
                    recvbufR=None

            comm.Gather(sendbufGLL,recvbufGLL,root=0)
            comm.Gather(sendbufGRR,recvbufGRR,root=0)
            comm.Gather(sendbufGLR,recvbufGLR,root=0)
            comm.Gather(sendbufGRL,recvbufGRL,root=0)
            if (parameters['Q']!=0):
                comm.Gather(sendbufQ,recvbufQ,root=0)
                comm.Gather(sendbufR,recvbufR,root=0)

        if (rank==0):
            fn_mu=('m'+str(parameters['mu']))*(parameters['x']!='delta0')
            fn_Delta='D'+str(parameters['delta0'])*(parameters['x']!='delta0')
            fn_alpha='a'+str(parameters['alpha_R'])*(parameters['x']!='delta0')
            fn_wl='L'+str(int(parameters['wireLength']))
            fn_muLead='muL'+str(parameters['muLead'])*(parameters['x']!='muLead')
            fn_bE=('bE'+str(parameters['barrierE']))*(parameters['x']!='barrierE')
            fn_potType=str(parameters['potType'])*(parameters['potType']!=0)
            fn_range=('-'+parameters['x']+'('+str(parameters['xMin'])+','+fn(parameters['xMax'])+')'+','+str(vBiasMax)+'-')
            # if parameters['muNum']==0:
            #     fn_range=('-'+str(vzStep*tot)+','+str(vBiasMax)+'-')*(parameters['muNum']==0)
            # else:
            #     fn_range=('-'+str(mu0)+','+str(mu0+muStep*tot)+','+str(vBiasMax)+'-')*(parameters['muNum']!=0)
            fn_potPeak=('mx'+str(parameters['potPeak']))*(parameters['potType']!=0)*(parameters['x']!='potPeak')
            fn_potPeakPos=('pk'+str(parameters['potPeakPos']))*((parameters['potType']=='lorentz')+( parameters['potType']=='lorentzsigmoid')+(parameters['potType']=='exp2'))*(parameters['x']!='potPeakPos')
            fn_potSigma=('sg'+str(parameters['potSigma']))*((parameters['potType']=='exp')+(parameters['potType']=='sigmoid')+(parameters['potType']=='exp2')+(parameters['potType']=='cos')+(parameters['potType']=='cos2'))*(parameters['x']!='potSigma')
            fn_potPeakR=('mxR'+str(parameters['potPeakR']))*(parameters['potType']=='exp2')*(parameters['x']!='potPeakR')
            fn_potPeakPosR=('pkR'+str(parameters['potPeakPosR']))*( parameters['potType']=='exp2')*(parameters['x']!='potPeakPosR')
            fn_potSigmaR=('sgR'+str(parameters['potSigmaR']))*(parameters['potType']=='exp2')*(parameters['x']!='potSigmaR')
            fn_muVar=('mVar'+str(parameters['muVar']))*(parameters['muVar']!=0)
            fn_muVarType=('C')*(parameters['muVarType']=='Coulomb')
            fn_dissipation=('G'+str(parameters['dissipation']))*(parameters['x']!='dissipation')
            fn_qdLength=('dL'+str(int(parameters['qdLength'])))*(parameters['isQD']!=0)*(parameters['x']!='qdLength')
            fn_qdPeak=('VD'+str(parameters['qdPeak']))*(parameters['isQD']!=0)*(parameters['x']!='qdPeak')
            fn_qdLengthR=('dLR'+str(int(parameters['qdLengthR'])))*(parameters['isQD']!=0)*(parameters['qdLengthR']!=0)*(parameters['x']!='qdLengthR')
            fn_qdPeakR=('VDR'+str(parameters['qdPeakR']))*(parameters['isQD']!=0)*(parameters['qdLengthR']!=0)*(parameters['x']!='qdPeakR')
            fn_couplingSCSM=('g'+str(parameters['couplingSCSM']))*(parameters['isSE']==1)*(parameters['x']!='couplingSCSM')
            fn_vc=('vc'+str(parameters['vc']))*(parameters['isSE']==1)*(parameters['vc']!=0)*(parameters['x']!='vc')
            fn_gVar=('gVar'+str(parameters['gVar']))*(parameters['gVar']!=0)
            fn_deltaVar=('DVar'+str(parameters['deltaVar']))*(parameters['deltaVar']!=0)
            fn_vz=('vz'+str(parameters['vz']))*(parameters['x']!='vz')

            # fnLL=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_potPeak+fn_potPeakPos+fn_potSigma+fn_potPeakR+fn_potPeakPosR+fn_potSigmaR+fn_muVarType+fn_muVar+fn_qdPeak+fn_qdLength+fn_qdPeakR+fn_qdLengthR+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+fn_bE+'LL'+fn_range
            # fnRR=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_potPeak+fn_potPeakPos+fn_potSigma+fn_potPeakR+fn_potPeakPosR+fn_potSigmaR+fn_muVarType+fn_muVar+fn_qdPeak+fn_qdLength+fn_qdPeakR+fn_qdLengthR+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+fn_bE+'RR'+fn_range
            # fnLR=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_potPeak+fn_potPeakPos+fn_potSigma+fn_potPeakR+fn_potPeakPosR+fn_potSigmaR+fn_muVarType+fn_muVar+fn_qdPeak+fn_qdLength+fn_qdPeakR+fn_qdLengthR+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+fn_bE+'LR'+fn_range
            # fnRL=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_potPeak+fn_potPeakPos+fn_potSigma+fn_potPeakR+fn_potPeakPosR+fn_potSigmaR+fn_muVarType+fn_muVar+fn_qdPeak+fn_qdLength+fn_qdPeakR+fn_qdLengthR+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+fn_bE+'RL'+fn_range

            fn=fn_vz+fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_muLead+fn_potType+fn_potPeak+fn_potPeakPos+fn_potSigma+fn_potPeakR+fn_potPeakPosR+fn_potSigmaR+fn_muVarType+fn_muVar+fn_qdPeak+fn_qdLength+fn_qdPeakR+fn_qdLengthR+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+fn_bE+fn_range
            fnLL=fn+'ll'
            fnRR=fn+'RR'
            fnLR=fn+'LR'
            fnRL=fn+'RL'

            np.savetxt(fnLL+'.dat',recvbufGLL)
            np.savetxt(fnRR+'.dat',recvbufGRR)
            np.savetxt(fnLR+'.dat',recvbufGLR)
            np.savetxt(fnRL+'.dat',recvbufGRL)
            if (parameters['Q']!=0):
                np.savetxt(fn+'Q.dat',recvbufQ)
                np.savetxt(fn+'R.dat',recvbufR)

            # if parameters['muNum']==0:
            #     xRange=np.arange(tot)*vzStep
            # else:
            #     xRange=mu0+np.arange(tot)*muStep
            xRange=np.linspace(parameters['xMin'],parameters['xMax'],tot)
            figLL=plt.figure()
            plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbufGLL), cmap='rainbow')
            # if parameters['muNum']==0:
            #     plt.xlabel('Vz(meV)')
            # else:
            #     plt.xlabel('mu(meV)')
            plt.xlabel(parameters['x']+'('+parameters['xUnit']+')')
            plt.ylabel(r'$V_\mathrm{bias}$ (meV)')
            plt.colorbar()
            plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax))
            figLL.savefig(fnLL+'.png')

            figRR=plt.figure()
            plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbufGRR), cmap='rainbow')
            # if parameters['muNum']==0:
            #     plt.xlabel('Vz(meV)')
            # else:
            #     plt.xlabel('mu(meV)')
            plt.xlabel(parameters['x']+'('+parameters['xUnit']+')')
            plt.ylabel(r'$V_\mathrm{bias}$ (meV)')
            plt.colorbar()
            plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax))
            figRR.savefig(fnRR+'.png')

            figLR=plt.figure()
            plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbufGLR), cmap='rainbow')
            # if parameters['muNum']==0:
            #     plt.xlabel('Vz(meV)')
            # else:
            #     plt.xlabel('mu(meV)')
            plt.xlabel(parameters['x']+'('+parameters['xUnit']+')')
            plt.ylabel(r'$V_\mathrm{bias}$ (meV)')
            plt.colorbar()
            plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax))
            figLR.savefig(fnLR+'.png')

            figRL=plt.figure()
            plt.pcolormesh(xRange,vBiasRange,np.transpose(recvbufGRL), cmap='rainbow')
            # if parameters['muNum']==0:
            #     plt.xlabel('Vz(meV)')
            # else:
            #     plt.xlabel('mu(meV)')
            plt.xlabel(parameters['x']+'('+parameters['xUnit']+')')
            plt.ylabel(r'$V_\mathrm{bias}$ (meV)')
            plt.colorbar()
            plt.axis((xRange[0],xRange[-1],vBiasMin,vBiasMax))
            figRL.savefig(fnRL+'.png')

            if (parameters['Q']==1):
                figQ=plt.figure()
                plt.plot(xRange,recvbufQ)
                plt.xlabel('Vz(meV)')
                plt.ylabel('det(r)')
                plt.axis((xRange[0],xRange[-1],-1,1))
                figQ.savefig(fn+'Q.png')


if __name__=="__main__":
    #start=time.time()
    main()
    #end=time.time()
    #print(end-start)
