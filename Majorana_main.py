import matplotlib
matplotlib.use('Agg')
from mpi4py.futures import MPIPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from Majorana_utils import Nanowire
import time
import random
import sys
from collections import defaultdict,OrderedDict
from itertools import repeat
   
def parse_arguments(parser,inputs=None):
    '''
    Parse input arguments for the nanowire parameters. The detailed description can be invoked by `python Majorana_main.py -h`

    Parameters
    ----------
    parser : argparse.ArgumentParser
            The input arguments for the nanowire.

    Returns
    -------
    args : targparse.ArgumentParser
            Arguments after `add_argument`.
    '''
    parser.add_argument('-mass','--mass',default=0.01519,type=float,help='effective mass (m_e)')
    parser.add_argument('-a','--a',default=10,type=float,help='lattice constant (nm)')
    parser.add_argument('-mu','--mu',default=0.2,type=float,help='chemical potential in the nanowire with respect to the band bottom (meV)')
    parser.add_argument('-alpha','--alpha',default=0.5,type=float,help='spin orbit coupling (eV.A)')
    parser.add_argument('-Delta0','--Delta0',default=0.2,type=float,help='superconducting gap (meV)')
    parser.add_argument('-L','--L',default=10,type=float,help='total length (um)')
    parser.add_argument('-mu_lead','--mu_lead',default=25,type=float,help='chemical potential in th"e lead (meV)')
    parser.add_argument('-barrier_num','--barrier_num',default=2,type=int,help='number of cells for the barrier')
    parser.add_argument('-barrier_E','--barrier_E',default=10,type=float,help='barrier height')
    parser.add_argument('-dissipation','--dissipation',default=0.0001,type=float,help='dissipation (meV)')
    parser.add_argument('-barrier_relative','--barrier_relative',default=None,type=float,help='barrier height relative to the chemical potential in the nanowire (meV)')
    # quantum dots (QD)
    parser.add_argument('-QD','--QD',action='store_true',help='flag for the quantum dot')
    parser.add_argument('-QD_peak','--QD_peak',default=0.4,type=float,help='peak height of the (left) quantum dot (meV)')
    parser.add_argument('-QD_L','--QD_L',default=0.2,type=float,help='length of the (left) quantum dot (um)')
    parser.add_argument('-QD_peak_R','--QD_peak_R',default=0,type=float,help='peak height of the right quantum dot (meV)')
    parser.add_argument('-QD_L_R','--QD_L_R',default=0,type=float,help='length of the right quantum dot (um)')
    # self-energy (SE)
    parser.add_argument('-SE','--SE',action='store_true',help='flag for the self-energy')
    parser.add_argument('-coupling_SC_SM','--coupling_SC_SM',default=0.2,type=float,help='coupling between the SC and SM (meV)')
    parser.add_argument('-Vzc','--Vzc',default=float('inf'),type=float,help='criticial Vz for SC gap collapsing (meV)')
    # inhomogeneous potential
    parser.add_argument('-pot_type','--pot_type',default='0',type=str,help='type of inhomogeneous potential (see README.MD for the detailed definition)')
    parser.add_argument('-pot_peak_pos','--pot_peak_pos',default=0,type=float,help='the location of the (left) peak of inhomogeneous potential (um)')
    parser.add_argument('-pot_sigma','--pot_sigma',default=1,type=float,help='the linewith of the (left) peak potential (um)')
    parser.add_argument('-pot_peak','--pot_peak',default=0,type=float,help='the peak height of the (left) quantum dot (meV)')
    parser.add_argument('-pot_peak_pos_R','--pot_peak_pos_R',default=0,type=float,help='the location of the right peak of inhomogeneous potential (um)')
    parser.add_argument('-pot_sigma_R','--pot_sigma_R',default=0,type=float,help='the linewith of the right peak potential (um)')
    parser.add_argument('-pot_peak_R','--pot_peak_R',default=0,type=float,help='the peak height of the right quantum dot (meV)')
    # disorder in chemical potential
    parser.add_argument('-muVar','--muVar',default=0,type=float,help='disorder strength in the chemical potential in SM (meV)')
    parser.add_argument('-muVar_fn','--muVar_fn',default='',type=str,help='the filename for the random list of the chemical potential in SM')
    parser.add_argument('-muVar_seed','--muVar_seed',default=None,type=int,help='the seed for the random list of the chemical potential in SM')
    # disorder in orther parameters
    parser.add_argument('-gVar','--gVar',default=0,type=float,help='disorder strength in g factor')
    parser.add_argument('-DeltaVar','--DeltaVar',default=0,type=float,help='disorder strength in SC gap (meV)')
    parser.add_argument('-coupling_SC_SM_Var','--coupling_SC_SM_Var',default=0,type=float,help='disorder strength in coupling strength between SC and SM (meV)')
    parser.add_argument('-random_fn','--random_fn',default='',type=str,help='filename for the random list (for g factor, Delta, and the coupling of SC & SM)')
    parser.add_argument('-random_seed','--random_seed',default=None,type=int,help='random seed for the random list (for g factor, Delta, and the coupling of SC & SM)')
    # lead
    parser.add_argument('-lead_pos','--lead_pos',default='L',type=str,choices=['L','R','LR','RL'],help='position of lead')   # for lead in lead_pos
    parser.add_argument('-lead_num','--lead_num',default=1,type=int,choices=[1,2],help='number of leads')
    # plot conductance spectrum
    parser.add_argument('-x','--x',default='Vz',type=str,help='scan of x-axis')
    parser.add_argument('-Vz','--Vz',default=0,type=float,help='Zeeman field (meV)')
    parser.add_argument('-x_min','--x_min',default=0,type=float,help='the min of x-axis')
    parser.add_argument('-x_max','--x_max',default=2.048,type=float,help='the max of x-axis')
    parser.add_argument('-x_num','--x_num',default=256,type=int,help='the number of points in x-axis')
    parser.add_argument('-y','--y',default='V_bias',type=str,help='scan of y-axis')
    parser.add_argument('-V_bias','--V_bias',default=0,type=float,help='Bias voltage (meV)')
    parser.add_argument('-x_unit','--x_unit',default='meV',type=str,help='the unit of x-axis')
    parser.add_argument('-y_min','--y_min',default=-0.3,type=float,help='the min of y-axis')
    parser.add_argument('-y_max','--y_max',default=0.3,type=float,help='the max of y-axis')
    parser.add_argument('-y_num','--y_num',default=301,type=int,help='the number of points in y-axis')
    parser.add_argument('-y_unit','--y_unit',default='mV',type=str,help='the unit of y-axis')
    parser.add_argument('-cmap','--cmap',default='seismic',type=str,help='cmap for the conductance spectrum')
    parser.add_argument('-vmin','--vmin',default=0,type=float,help='the min of conductance for the color bar')
    parser.add_argument('-vmax','--vmax',default=4,type=float,help='the max of conductance for the color bar')
    # crossover
    # parser.add_argument('-alpha','--alpha',default=-1,type=float,help='crossover parameter')
    # Calculation type
    parser.add_argument('-conductance','--conductance',action='store_true',help='flag for calculating the conductance spectrum (default True)')
    parser.add_argument('-LDOS','--LDOS',action='store_true',help='flag for calculating the LDOS (default False)')
    parser.add_argument('-wavefunction','--wavefunction',action='store_true',help='flag for calculating the wavefunction (default False)')
    if inputs is None:
        args=parser.parse_args()
    else:
        args=parser.parse_args(inputs)
    args.muVar_seed=random.randrange(sys.maxsize) if args.muVar_seed is None else args.muVar_seed
    args.random_seed=random.randrange(sys.maxsize) if args.random_seed is None else args.random_seed
    return args

def wrapper(inputs):
    args,x,y=inputs
    nw=Nanowire(args)
    G,TVL,TVR,kappa=nw.conductance(x,y) if args.conductance else repeat(None,4)
    if args.LDOS:
        assert args.y=='V_bias', "y has to be v_bias to calculate LDOS" 
        LDOS=nw.LDOS(x,y)
    else:
        LDOS=None

    if args.wavefunction:
        assert args.y=='V_bias', "y has to be v_bias to calculate wavefunction" 
        nw.wavefunction(x,y)
    
    return [G,TVL,TVR,kappa,LDOS]

def postprocess_G(G_raw):
    
    return {lead_pos:np.array([G[lead_pos] for G in G_raw]).reshape((args.x_num,args.y_num)) for lead_pos in G_raw[0].keys()} if args.conductance else None

def postprocess_S(S_raw):
    return np.array([S for S in S_raw if S is not None]) if args.conductance else None

def postprocess_LDOS(LDOS_raw):
    return np.array(list(LDOS_raw)).reshape((args.x_num,args.y_num,-1)) if args.LDOS else None

def filename(args):
    fn=OrderedDict()
    fn['Vz']='Vz{}'.format(args.Vz)
    fn['mu']='m{}'.format(args.mu)
    fn['Delta0']='D{}'.format(args.Delta0)
    fn['DeltaVar']='DVar{}'.format(args.DeltaVar) if args.DeltaVar>0 else ''
    fn['alpha']='a{}'.format(args.alpha)
    fn['L']='L{}'.format(args.L)
    fn['mu_lead']='mL{}'.format(args.mu_lead)
    if args.pot_type!='0':
        fn['pot_type']='{}'.format(args.pot_type)
        fn['pot_peak']='mx{}'.format(args.pot_peak) if args.pot_peak_R==0 else 'mx{},{}'.format(args.pot_peak,args.pot_peak_R)
        fn['pot_peak_pos']='pk{}'.format(args.pot_peak_pos) if args.pot_peak_pos_R==0 else 'pp{},{}'.format(args.pot_peak_pos,args.pot_peak_pos_R)
        fn['pot_sigma']='sg{}'.format(args.pot_peak_sigma) if args.pot_sigma_R==0 else 'sg{},{}'.format(args.pot_peak_sigma,args.pot_peak_sigma_R)
    fn['muVar']='mVar{}'.format(args.muVar) if args.muVar>0 else ''
    if args.QD:
        fn['QD_L']='dL{}'.format(args.QD_L) if args.QD_L_R==0 else 'dL{},{}'.format(args.QD_L,args.QD_L_R)
        fn['QD_peak']='VD{}'.format(args.QD_peak) if args.QD_peak_R==0 else 'VD{},{}'.format(args.QD_peak,args.QD_peak_R)
    if args.SE:
        fn['coupling_SC_SM']='g{}'.format(args.coupling_SC_SM)
        fn['coupling_SC_SM_Var']='gammaVar{}'.format(coupling_SC_SM_Var) if args.coupling_SC_SM_Var>0 else ''
        fn['Vzc']='Vzc{}'.format(args.Vzc) if args.Vzc<float('inf') else ''
    
    fn['gVar']='gVar{}'.format(args.gVar) if args.gVar>0 else ''
    fn['dissipation']='G{}'.format(args.dissipation) if args.dissipation>0 else ''
    fn['barrier_E']='' if args.barrier_relative is None else 'bE{}'.format(args.barrier_E)
    fn['barrier_E']='bE{}'.format(args.barrier_E) if args.barrier_relative is None else 'bR{}'.format(args.barrier_relative)
    fn['range']='-{}({},{}),{}({},{})-'.format(args.x,args.x_min,args.x_max,args.y,args.y_min,args.y_max)
    fn['lead_pos']='{}'.format(args.lead_pos)
    fn[args.x]=''
    fn[args.y]=''
    return ''.join(fn.values())


def plot(fn):
    if args.conductance:
        if len(G.keys())==1:
            fig,ax=plt.subplots(tight_layout=True)
            for key, value in G.items():
                im=ax.pcolormesh(x_range,y_range,value.T,cmap=args.cmap,vmin=args.vmin,vmax=args.vmax,shading='auto')
                axins=ax.inset_axes([1.02,0,.05,1],transform=ax.transAxes)
                cb=plt.colorbar(im,cax=axins)
                cb.ax.set_title(r'$G(e^2/h)$')
                ax.text(.5,1,key,transform=ax.transAxes,va='bottom',ha='center')
            ax.set_xlabel('{}({})'.format(args.x,args.x_unit))
            ax.set_ylabel('{}({})'.format(args.y,args.y_unit))
        elif len(G.keys())==2:
            fig,axs=plt.subplots(1,2,tight_layout=True)
            for ax,key in zip(axs,['L','R']):
                value=G[key]
                im=ax.pcolormesh(x_range,y_range,value.T,cmap=args.cmap,vmin=args.vmin,vmax=args.vmax,shading='auto')
                axins=ax.inset_axes([1.02,0,.05,1],transform=ax.transAxes)
                cb=plt.colorbar(im,cax=axins)
                cb.ax.set_title(r'$G(e^2/h)$')
                ax.text(.5,1,key,transform=ax.transAxes,va='bottom',ha='center')
                ax.set_xlabel('{}({})'.format(args.x,args.x_unit))
            axs[0].set_ylabel('{}({})'.format(args.y,args.y_unit))
        elif len(G.keys())==4:
            fig,axs=plt.subplots(3,2,tight_layout=True)
            for ax,key in zip(axs.flatten()[:4],['LL','RR','LR','RL']):
                value=G[key]
                im=ax.pcolormesh(x_range,y_range,value.T,cmap=args.cmap,vmin=args.vmin if key in ['LL','RR'] else None,vmax=args.vmax if key in ['LL','RR'] else None,shading='auto')
                axins=ax.inset_axes([1.02,0,.05,1],transform=ax.transAxes)
                cb=plt.colorbar(im,cax=axins)
                cb.ax.set_title(r'$G(e^2/h)$')
                ax.text(.5,1,key,transform=ax.transAxes,va='bottom',ha='center')
            [ax.set_xlabel('{}({})'.format(args.x,args.x_unit)) for ax in axs[-1,:]]
            [ax.set_ylabel('{}({})'.format(args.y,args.y_unit)) for ax in axs[:,0]]
            axs[2,0].plot(x_range,TVL,label='L',color='r')
            axs[2,0].plot(x_range,TVR,label='R',color='b')
            axs[2,0].set_ylabel('TV')
            axs[2,0].legend()
            axs[2,1].plot(x_range,kappa)
            axs[2,1].set_ylabel(r'$\kappa/\kappa_0$')
        fig.savefig('{}_cond.png'.format(fn),bbox_inches='tight')
    if args.LDOS:
        pass

def savedata(fn):
    data={}
    data['args']=args
    if args.conductance:
        data['G']=G
        data['TVL']=TVL
        data['TVR']=TVR
        data['kappa']=kappa
    if args.LDOS:
        data['LDOS']=LDOS
    if args.wavefunction:
        pass

    with open('{}.pickle'.format(fn),'wb') as f:
        pickle.dump(data,f)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    args=parse_arguments(parser)
    print(args)
    # nw.get_potential_type()

    st=time.time()
    
    x_range=np.linspace(args.x_min, args.x_max,args.x_num)
    y_range=np.linspace(args.y_min, args.y_max,args.y_num)
    inputs=[(args,x,y) for x in x_range for y in y_range]
    
    # with MPIPoolExecutor() as executor:
    #     rs=list(executor.map(wrapper,inputs))
    rs=list(map(wrapper,inputs))

    G_raw,TVL_raw,TVR_raw,kappa_raw,LDOS_raw=zip(*rs)
    G=postprocess_G(G_raw)
    TVL=postprocess_S(TVL_raw)
    TVR=postprocess_S(TVR_raw)
    kappa=postprocess_S(kappa_raw)
    LDOS=postprocess_LDOS(LDOS_raw)


    fn=filename(args)
    plot(fn=fn)
    savedata(fn=fn)
    
    print(time.time()-st)


