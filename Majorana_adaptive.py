import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
import numpy as np
import Majorana_module as Maj
import sys
import re
import matplotlib.pyplot as plt
import adaptive
from scipy import interpolate
from mpi4py.futures import MPIPoolExecutor
from math import sqrt
from functools import partial
import time
from copy import deepcopy


def main():
    start=time.time()
    vars=len(sys.argv);    
    parameters = {'isTV':0,'a':1,'mu':.2,'alpha_R':5, 'delta0':0.2,'wireLength':1000,
               'muLead':25.0, 'barrierNum':2,'barrierE':10.0, 'dissipation':0.0001,'isDissipationVar':0, 
               'isQD':0, 'qdPeak':0.4, 'qdLength':20, 
               'isSE':0, 'couplingSCSM':0.2, 'vc':0,               
               'potType':0,'potPeakPos':0,'potSigma':1,'potPeak':0,
               'muVar':0,'muVarList':0,
               'gVar':0,'randList':0,
               'deltaVar':0,
               'vz':0.0,'vz0':0, 'vBias':0.0,'vBiasMin':-0.3,'vBiasMax':0.3,'vzNum':256,'vBiasNum':1001,'vzStep': 0.002,'vzMax':1.024,'mu0':0,'muMax':1,'muStep':0.002,'isMu':0,
               'leadPos':0,'leadNum':1,     
               'loss':0.01,
               'error':0};
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
                    sys.exit(1);
            except:
                print('Cannot parse the input parameters',sys.argv[i]);
                parameters['error']=1;
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
                sys.exit(1);
        except:
            print('Cannot find disorder file:',muVarfn);
            parameters['error']=1;
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
                sys.exit(1);
        except:
            print('Cannot find random list file:',randfn);
            parameters['error']=1;
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
    
        
    if parameters['error']!=0:
        sys.exit(1);

    np.warnings.filterwarnings('ignore')
    vBiasMin = parameters['vBiasMin']; 
    vBiasMax = parameters['vBiasMax']; 
    vBiasNumber = int(parameters['vBiasNum']);
    vBiasRange = np.linspace(vBiasMin, vBiasMax, vBiasNumber);    
    #randList=parameters['randList'];    
    if parameters['leadNum']==1:
        cond_partial=partial(cond,parameters=parameters)
        leadPos=int(parameters['leadPos']);
        for irun in range(leadPos+1):
            parameters['leadPos']=irun;
            lossfunc=adaptive.learner.learner2D.resolution_loss_function(min_distance=0.001,max_distance=1)
            if parameters['isMu']==0:
                learner=adaptive.Learner2D(cond_partial,bounds=[(parameters['vz0'],parameters['vzMax']),(parameters['vBiasMin'],parameters['vBiasMax'])]);
                #,loss_per_triangle=adaptive.learner.learner2D.minimize_triangle_surface_loss
            else:
                learner=adaptive.Learner2D(cond_partial,bounds=[(parameters['mu0'],parameters['muMax']),(parameters['vBiasMin'],parameters['vBiasMax'])]);
            runner=adaptive.BlockingRunner(learner,goal=lambda l:l.loss()<parameters['loss'],executor=MPIPoolExecutor(),shutdown_executor=True)
            save_fig(learner,n=501,parameters=parameters)
    elif parameters['leadNum']==2:
        cond_matrix_partial=partial(cond_matrix,parameters=parameters)
        lossfunc=adaptive.learner.learner2D.resolution_loss_function(min_distance=0.0,max_distance=1)
        if parameters['isMu']==0:
            learner=adaptive.Learner2D(cond_matrix_partial,bounds=[(parameters['vz0'],parameters['vzMax']),(parameters['vBiasMin'],parameters['vBiasMax'])]);
        else:
            learner=adaptive.Learner2D(cond_matrix_partial,bounds=[(parameters['mu0'],parameters['muMax']),(parameters['vBiasMin'],parameters['vBiasMax'])]);
        runner=adaptive.BlockingRunner(learner,goal=lambda l:l.loss()<parameters['loss'],executor=MPIPoolExecutor(),shutdown_executor=True)
        save_fig(learner,n=501,parameters=parameters)
    end=time.time()
    print(end-start)
    print(len(learner.data))

def cond(xy,parameters=None):
    x,y=xy
    pm=deepcopy(parameters)
    if pm['isMu']==0:
        pm['vz']=x     
    else:
        pm['mu']=x
    pm['vBias']=y
    if pm['gVar']!=0:
        pm['randList']=pm['randList']*pm['vz'];

    junction=Maj.make_NS_junction(pm)  
    return Maj.conductance(pm,junction)  
 
def cond_matrix(xy,parameters=None):
    x,y=xy
    pm=deepcopy(parameters)
    if pm['isMu']==0:
        pm['vz']=x     
    else:
        pm['mu']=x
    pm['vBias']=y
    if pm['gVar']!=0:
        pm['randList']=pm['randList']*pm['vz'];

    junction=Maj.make_NS_junction(pm)  
    return Maj.conductance_matrix(pm,junction)    

def areas(ip):
    p=ip.tri.points[ip.tri.vertices]
    q=p[:,:-1,:]-p[:,-1,None,:]
    areas=abs(q[:,0,0]*q[:,1,1]-q[:,0,1]*q[:,1,0])/2
    return areas
    
def save_fig(self,n=None,parameters=None):
    xb,yb=self.bounds
    # lbrt=xb[0],yb[0],xb[1],yb[1]
    ip= self.ip()
    if n is None:
        # calculate how many grid points are needed.
        n= int(.658/sqrt(areas(ip).min()))
        n=max(n,10)
    eps=1e-13
    x=y=np.linspace(-.5+eps,.5-eps,n)
    z=ip(x[:,None],y[None,:]*self.aspect_ratio).squeeze()
    xRange=np.linspace(xb[0],xb[1],n)
    yRange=np.linspace(yb[0],yb[1],n)
     # lead1PosSet=("L","R")
    fn_mu=('m'+str(parameters['mu']))*(parameters['isMu']==0);
    fn_Delta='D'+str(parameters['delta0']);
    fn_alpha='a'+str(parameters['alpha_R']);
    fn_wl='L'+str(int(parameters['wireLength']));
    fn_potType=str(parameters['potType'])*(parameters['potType']!=0);
    fn_leadPos='L'*(parameters['leadPos']==0)+'R'*(parameters['leadPos']==1);
    if parameters['isMu']==0:
        fn_range=('-'+str(parameters['vz0'])+','+str(parameters['vzMax'])+','+str(parameters['vBiasMax'])+'-')*(parameters['isMu']==0);
    else:
        fn_range=('-'+str(parameters['mu0'])+','+str(parameters['muMax'])+','+str(parameters['vBiasMax'])+'-')*(parameters['isMu']!=0);  
    fn_potPeak=('mx'+str(parameters['potPeak']))*(parameters['potType']!=0);  
    fn_potPeakPos=('pk'+str(parameters['potPeakPos']))*((parameters['potType']=='lorentz')+( parameters['potType']=='lorentzsigmoid'));
    fn_potSigma=('sg'+str(parameters['potSigma']))*((parameters['potType']=='exp')+(parameters['potType']=='sigmoid'));
    fn_muVar=('mVar'+str(parameters['muVar']))*(parameters['muVar']!=0)    
    fn_dissipation=('G'+str(parameters['dissipation']))*(parameters['isDissipationVar']!=0);
    fn_qdLength=('dL'+str(int(parameters['qdLength'])))*(parameters['isQD']!=0);
    fn_qdPeak=('VD'+str(parameters['qdPeak']))*(parameters['isQD']!=0);
    fn_couplingSCSM=('g'+str(parameters['couplingSCSM']))*(parameters['isSE']==1);
    fn_vc=('vc'+str(parameters['vc']))*(parameters['isSE']==1)*(parameters['vc']!=0);
    fn_gVar=('gVar'+str(parameters['gVar']))*(parameters['gVar']!=0);
    fn_deltaVar=('DVar'+str(parameters['deltaVar']))*(parameters['deltaVar']!=0);   
    if parameters['leadNum']==1:
        fn=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_potPeak+fn_potPeakPos+fn_potSigma+fn_muVar+fn_qdPeak+fn_qdLength+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+fn_leadPos+fn_range;
        np.savetxt(fn+'.dat',z);
        fig=plt.figure()
        plt.pcolormesh(xRange,yRange,z.T,cmap='rainbow')
        if parameters['isMu']==0:            
            plt.xlabel('Vz(meV)');
        else:
            plt.xlabel('mu(meV)');
        plt.ylabel('V_bias(meV)');
        plt.colorbar();
        plt.axis((xb[0],xb[1],yb[0],yb[1]))
        fig.savefig(fn+'.png')
    else:
        fn_leadPosSet=("LL","RR","LR","RL")
        for i in range(z.shape[2]):
            fn=fn_mu+fn_Delta+fn_deltaVar+fn_alpha+fn_wl+fn_potType+fn_potPeak+fn_potPeakPos+fn_potSigma+fn_muVar+fn_qdPeak+fn_qdLength+fn_couplingSCSM+fn_vc+fn_dissipation+fn_gVar+fn_leadPosSet[i]+fn_range;
            np.savetxt(fn+'.dat',z[:,:,i])
            fig=plt.figure()
            plt.pcolormesh(xRange,yRange,z[:,:,i].T,cmap='rainbow')
            if parameters['isMu']==0:            
                plt.xlabel('Vz(meV)');
            else:
                plt.xlabel('mu(meV)');
            plt.ylabel('V_bias(meV)')
            plt.colorbar()        
            plt.axis((xb[0],xb[1],yb[0],yb[1]))
            fig.savefig(fn+'.png')
            
if __name__=="__main__":
	main()
