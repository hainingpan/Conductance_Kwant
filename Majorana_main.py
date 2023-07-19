import matplotlib
matplotlib.use('Agg')
from mpi4py.futures import MPIPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import pickle
from Majorana_utils import Nanowire
import time
import random
import sys
from collections import defaultdict,OrderedDict
from itertools import repeat
from scipy.signal import find_peaks
import warnings
from tqdm import tqdm
   
def parse_arguments(parser,args=None):
    '''
    Parse input arguments for the nanowire parameters. The detailed description can be invoked by `python Majorana_main.py -h`.

    Parameters
    ----------
    parser : argparse.ArgumentParser
            The uninitialized arguments for the nanowire.
    args : List of string or None
        If args==None, sys.argv[1:] is parsed. Otherwise, the `args` is parsed.

    Returns
    -------
    args : argparse.ArgumentParser
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
    parser.add_argument('-massVar','--massVar',default=0,type=float,help='disorder strength in effective mass (m_e)')
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
    parser.add_argument('-energy','--energy',action='store_true',help='flag for calculating the energy (default False)')
    parser.add_argument('-wavefunction','--wavefunction',action='store_true',help='flag for calculating the wavefunction (default False)')
    args=parser.parse_args(args)
    args.muVar_seed=random.randrange(sys.maxsize) if args.muVar_seed is None else args.muVar_seed
    args.random_seed=random.randrange(sys.maxsize) if args.random_seed is None else args.random_seed
    return args

def wrapper(inputs):
    '''
    Wrap all the functions and sent to parallel.
    
    Parameters
    ----------
    inputs : (argparse.ArgumentParser,float,float)
            Which point of (x,y) to be evaluated given the parameters in `args`.
    
    Returns
    -------
    List
        The list of [G,TV,kappa,LDOS,En] for conductance, topological visibility, thermal condutance, local density of states, energy spaectrum.
    '''
    args,x,y=inputs
    nw=Nanowire(args)
    G,TV,kappa=nw.conductance(x,y) if args.conductance else repeat(None,3)
    if args.LDOS:
        assert args.y=='V_bias', "y has to be v_bias to calculate LDOS" 
        LDOS=nw.LDOS(x,y)
    else:
        LDOS=None

    if args.energy:
        assert args.y=='V_bias', "y has to be v_bias to calculate LDOS."
        assert args.dissipation==0, "Disispation ({}) should be set to zero.".format(args.dissipation)
        if not args.barrier_E==0:
            warnings.warn(f'Tunnel ({args.barrier_E}) barrier should be 0.')
        if args.SE:
            if LDOS is None:
                LDOS=nw.LDOS(x,y)
            En = None
        else:
            En = nw.ED(x,y)
    else:
        En=None    

    if args.wavefunction:
        assert args.y=='V_bias', "y has to be v_bias to calculate LDOS."
        assert args.dissipation==0, "Disispation ({}) should be set to zero.".format(args.dissipation)
        assert args.barrier_E==0,'Tunnel ({}) barrier should be 0.'.format(args.barrier_E)
        nw.wavefunction(x,y)
    
    return [G,TV,kappa,LDOS,En]

def postprocess_G(G_raw):
    '''
    Postprocessing original conductance `G_raw` from the `wrapper`.
    
    Parameters
    ----------
    G_raw : tuple
            The tuple of conductance, where each element in the tuple is a dictionary containing 'L' or 'R' for left and right conductance, etc. 
    
    Returns
    -------
    dict 
        The dictionary for conductance, where keys are `lead_pos` and values are conductance spectrum of a np.array with the dimension of `x_num` and `y_num`.
    '''
    return {lead_pos:np.array([G[lead_pos] for G in G_raw]).reshape((args.x_num,args.y_num)) for lead_pos in G_raw[0].keys()} if args.conductance else None

def postprocess_S(S_raw):
    '''
    Postprocessing the original TV and thermal conductance `S_raw` from the `wrapper`. Note that these only are defined at zero bias. So None returned from `wrapper` should be ignored.
    
    Parameters
    ----------
    S_raw : tuple
            The tuple of conductance, where each element in the tuple is a dictionary containing 'L' or 'R' for left and right conductance, etc. 
    
    Returns
    -------
    dict 
        The dictionary for conductance, where keys are `lead_pos` and values are conductance spectrum of a np.array with the dimension of `x_num` and `y_num`.
    '''
    for S in S_raw:
        if S is not None:
            keys=S.keys()
            break
    else:
        return None
    return {lead_pos:np.array([S[lead_pos] for S in S_raw if S is not None]) for lead_pos in keys} if args.conductance else None

def postprocess_LDOS(LDOS_raw):
    '''
    Postprocessing the original LDOS from the `wrapper`.
    
    Parameters
    ----------
    LDOS_raw : tuple
            The tuple of LDOS, where each element in the tuple is a 1d array with `(wire_num,1)`.
    
    Returns
    -------
    np.array
            The 3D array of LDOS with the dimension of (`x_num`,`y_num`,`wire_num`).
    '''
    return np.array(list(LDOS_raw)).reshape((args.x_num,args.y_num,-1)) if args.LDOS or args.energy and args.SE else None

def postprocess_En(En_raw):
    '''
    Postprocessing the original energy spectrum from the `wrapper`. 
    
    Parameters
    ----------
    En_raw : tuple
            The tuple of energy spectrum, where each element in the tuple is a 1d array with `(4*wire_num,1)`.
    
    Returns
    -------
    dict
        The dict for the energy spectrum, where keys are x values and values are y values. 
    '''
    return {x:energy for x,energy in zip(np.linspace(args.x_min, args.x_max,args.x_num),np.array([En for En in En_raw if En is not None]))}

def filename(args):
    '''
    Generate file name.
    
    Returns
    -------
    string
        The filename to be exported. 
    '''
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
        fn['pot_sigma']='sg{}'.format(args.pot_sigma) if args.pot_sigma_R==0 else 'sg{},{}'.format(args.pot_sigma,args.pot_sigma_R)
    fn['muVar']='mVar{}'.format(args.muVar) if args.muVar>0 else ''
    if args.QD:
        fn['QD_L']='dL{}'.format(args.QD_L) if args.QD_L_R==0 else 'dL{},{}'.format(args.QD_L,args.QD_L_R)
        fn['QD_peak']='VD{}'.format(args.QD_peak) if args.QD_peak_R==0 else 'VD{},{}'.format(args.QD_peak,args.QD_peak_R)
    if args.SE:
        fn['coupling_SC_SM']='g{}'.format(args.coupling_SC_SM)
        fn['coupling_SC_SM_Var']='gammaVar{}'.format(coupling_SC_SM_Var) if args.coupling_SC_SM_Var>0 else ''
        fn['Vzc']='Vzc{}'.format(args.Vzc) if args.Vzc<float('inf') else ''
    
    fn['gVar']='gVar{}'.format(args.gVar) if args.gVar>0 else ''
    fn['massVar']='massVar{}'.format(args.massVar) if args.massVar>0 else ''
    fn['dissipation']='G{}'.format(args.dissipation) if args.dissipation>0 else ''
    fn['barrier_E']='' if args.barrier_relative is None else 'bE{}'.format(args.barrier_E)
    fn['barrier_E']='bE{}'.format(args.barrier_E) if args.barrier_relative is None else 'bR{}'.format(args.barrier_relative)
    fn['range']='-{}({},{}),{}({},{})-'.format(args.x,args.x_min,args.x_max,args.y,args.y_min,args.y_max)
    fn['lead']='{}{}'.format(args.lead_num,args.lead_pos)
    fn[args.x]=''
    fn[args.y]=''
    return ''.join(fn.values())

def plot_G_1(x_range,y_range,G,args):
    '''
    Plot the conductance in the nanowire with one lead.
    
    Parameters
    ----------
    x_range : np.array
            Range of x-axis, default is `x_min` to `x_max`.
    y_range : np.array
            Range of y-axis, default is `y_min` to `y_max`.
    G : dict
        Conductance with labels of 'L' or 'R'.
    args : argparse.ArgumentParser
            Arguments.
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of plotting the LDOS.
    '''

    fig,ax=plt.subplots(tight_layout=True,figsize=(6.8,4))
    for key, value in G.items():
        im=ax.pcolormesh(x_range,y_range,value.T,cmap=args.cmap,vmin=args.vmin,vmax=args.vmax,shading='auto',rasterized=True)
        axins=ax.inset_axes([1.02,0,.05,1],transform=ax.transAxes)
        cb=plt.colorbar(im,cax=axins)
        cb.ax.set_title(r'$G(e^2/h)$')
        ax.text(.5,1,key,transform=ax.transAxes,va='bottom',ha='center')
    ax.set_xlabel('{}({})'.format(args.x,args.x_unit))
    ax.set_ylabel('{}({})'.format(args.y,args.y_unit))
    return fig

def plot_G_2(x_range,y_range,G,args):
    '''
    Plot the conductance in the nanowire with one lead but with both sides alternatively.
    
    Parameters
    ----------
    x_range : np.array
            Range of x-axis, default is `x_min` to `x_max`.
    y_range : np.array
            Range of y-axis, default is `y_min` to `y_max`.
    G : dict
        Conductance with labels of 'L'/'R'.
    args : argparse.ArgumentParser
            Arguments.
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of plotting the LDOS.
    '''
    fig,axs=plt.subplots(1,2,tight_layout=True,figsize=(6.8*2,4))
    for ax,key in zip(axs,['L','R']):
        value=G[key]
        im=ax.pcolormesh(x_range,y_range,value.T,cmap=args.cmap,vmin=args.vmin,vmax=args.vmax,shading='auto',rasterized=True)
        axins=ax.inset_axes([1.02,0,.05,1],transform=ax.transAxes)
        cb=plt.colorbar(im,cax=axins)
        cb.ax.set_title(r'$G(e^2/h)$')
        ax.text(.5,1,key,transform=ax.transAxes,va='bottom',ha='center')
        ax.set_xlabel('{}({})'.format(args.x,args.x_unit))
    axs[0].set_ylabel('{}({})'.format(args.y,args.y_unit))
    return fig

def plot_G_4(x_range,y_range,G,args):
    '''
    Plot the conductance in the nanowire with 2 leads.
    
    Parameters
    ----------
    x_range : np.array
            Range of x-axis, default is `x_min` to `x_max`.
    y_range : np.array
            Range of y-axis, default is `y_min` to `y_max`.
    G : dict
        Conductance with labels of 'LL'/'RR'/'LR'/'RL'.
    args : argparse.ArgumentParser
            Arguments.
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of plotting the LDOS.
    '''
    fig,axs=plt.subplots(3,2,tight_layout=True,figsize=(6.8*2,4*3))
    for ax,key in zip(axs.flatten()[:4],['LL','RR','LR','RL']):
        value=G[key]
        im=ax.pcolormesh(x_range,y_range,value.T,cmap=args.cmap,vmin=args.vmin if key in ['LL','RR'] else None,vmax=args.vmax if key in ['LL','RR'] else None,shading='auto',rasterized=True)
        axins=ax.inset_axes([1.02,0,.05,1],transform=ax.transAxes)
        cb=plt.colorbar(im,cax=axins)
        cb.ax.set_title(r'$G(e^2/h)$')
        ax.text(.5,1,key,transform=ax.transAxes,va='bottom',ha='center')
    [ax.set_xlabel('{}({})'.format(args.x,args.x_unit)) for ax in axs[-1,:]]
    [ax.set_ylabel('{}({})'.format(args.y,args.y_unit)) for ax in axs[:,0]]
    axs[2,0].plot(x_range,TV['L'],label='L',color='r')
    axs[2,0].plot(x_range,TV['R'],label='R',color='b')
    axs[2,0].set_ylabel('TV')
    axs[2,0].legend()
    axs[2,1].plot(x_range,kappa['LR'],label='LR',color='r')
    axs[2,1].plot(x_range,kappa['RL'],label='RL',color='b')
    axs[2,1].set_ylabel(r'$\kappa/\kappa_0$')
    axs[2,1].legend()
    return fig

def plot_LDOS(x_range,y_range,LDOS,args):
    '''
    Plot LDOS
    
    Parameters
    ----------
    x_range : np.array
            Range of x-axis, default is `x_min` to `x_max`.
    y_range : np.array
            Range of y-axis, default is `y_min` to `y_max`.
    LDOS : np.array
            LDOS in 2D array with (x_num,y_num).
    args : argparse.ArgumentParser
            Arguments.
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of plotting the LDOS.
    '''
    fig,ax=plt.subplots(tight_layout=True,figsize=(6.8,4))
    DOS=LDOS.sum(axis=-1)
    im=ax.pcolormesh(x_range,y_range,DOS.T,cmap='inferno',shading='auto',rasterized=True,norm=colors.LogNorm(vmin=DOS.min(), vmax=DOS.max()))
    axins=ax.inset_axes([1.02,0,.05,1],transform=ax.transAxes)
    cb=plt.colorbar(im,cax=axins)
    cb.ax.set_title(r'DOS')
    ax.set_xlabel('{}({})'.format(args.x,args.x_unit))
    ax.set_ylabel('{}({})'.format(args.y,args.y_unit))
    return fig

def plot_energy(energies,args):
    '''
    Plot the energy spectrum
    
    Parameters
    ----------
    energies : dict
            The dict for the energy spectrum, where keys are x values and values are y values. 
    
    Returns
    -------
    fig : matplotlib.figure.Figure
            Figure of plotting the energy spectrum.
    '''
    fig,ax=plt.subplots(tight_layout=True,figsize=(6.8,4))
    energy_pts=np.vstack([np.array([key*np.ones_like(val),val]).T for key,val in energies.items()])
    ax.scatter(*energy_pts.T,color='k',marker='.',s=5)
    ax.set_xlim([args.x_min,args.x_max])
    ax.set_ylim([args.y_min,args.y_max])
    ax.set_xlabel('{}({})'.format(args.x,args.x_unit))
    ax.set_ylabel('{}({})'.format(args.y,args.y_unit))
    return fig


def plot_wavefunction(result,args,fig=None,ax=None):
    '''
    Plot the wave function of BdG Hamiltonian, and Majorana basis, respectively.
    
    Parameters
    ----------
    args : argparse.ArgumentParser
            Inlucding this argument because args.L is needed.
    fig : matplotlib.figure.Figure, default=None
            Create a fig if not provided
    ax : matplotlib.axes._subplots.AxesSubplot
            Create a ax if not provided
    
    Returns
    -------
    fig : matplotlib.figure.Figure
            fig
    ax : matplotlib.axes._subplots.AxesSubplot
            ax
    '''
    wire=np.linspace(0,args.L,result['wf_p'].shape[0])
    if fig is None and ax is None:
        fig,ax=plt.subplots()
    ax.plot(wire,result['wf_p'],'k',label='$|\Psi|^2$')
    ax.plot(wire,result['wf_1'],'r',label='$|\gamma_1|^2$')
    ax.plot(wire,result['wf_2'],'b',label='$|\gamma_2|^2$')
    ax.set_title('E={:.5f}\n$E_{{trial}}$={:.5f}\n$\Delta E$={:e}'.format(result['val_p'],result['ansatz'],result['ansatz']-result['val_p']))
    ax.legend()
    ax.set_xlabel('L ($\mu$m)')
    return fig,ax

def plot(fn):
    '''
    Plot the conductance, LDOS, energy spectrums, and save as png.
    
    Parameters
    ----------
    fn : str
            The filename for pickle file. 
    '''
    if args.conductance:
        if len(G.keys())==1:
            fig=plot_G_1(x_range, y_range, G, args)
        elif len(G.keys())==2:
            fig=plot_G_2(x_range, y_range, G, args)
        elif len(G.keys())==4:
            fig=plot_G_4(x_range, y_range, G, args)
        fig.savefig('{}_cond.png'.format(fn),bbox_inches='tight',dpi=1000)
    if args.LDOS:
        fig=plot_LDOS(x_range, y_range, LDOS, args)
        fig.savefig('{}_LDOS.png'.format(fn),bbox_inches='tight',dpi=1000)
    if args.energy:
        fig=plot_energy(energies, args)
        fig.savefig('{}_energy.png'.format(fn),bbox_inches='tight',dpi=1000)


def savedata(fn):
    '''
    Dump all data into pickle.
    
    Parameters
    ----------
    fn : str
        The filename for pickle file.
    '''
    data={}
    data['args']=args
    if args.conductance:
        data['G']=G
        data['TV']=TV
        data['kappa']=kappa
    if args.LDOS:
        data['LDOS']=LDOS
    if args.wavefunction:
        pass
    if args.energy:
        data['energies']=energies

    with open('{}.pickle'.format(fn),'wb') as f:
        pickle.dump(data,f)

def detect_peaks(LDOS):
    '''
    Detect peaks to find the divergence of the LDOS from the green's function.
    
    Parameters
    ----------
    LDOS : type
            LDOS
    
    Returns
    -------
    energies : dict
            Dict for the energy spectrum which are abstracted from the peaks, where the keys are `x` and the values are `y` (`V_bias`).
    '''
    assert args.SE, 'Use ED for systems without self-energy'
    energies={}
    DOS=LDOS.sum(axis=-1)
    vbias_list=np.linspace(args.y_min,args.y_max,args.y_num)
    assert args.x=='Vz', 'x axis only supports Vz'
    x_list=np.linspace(args.x_min,args.x_max,args.x_num)
    for ind in range(DOS.shape[0]):
        x=x_list[ind]
        vbias_max=args.Delta0*np.sqrt(1-(x/args.Vzc)**2)
        DOS_line=DOS[ind,:]
        pks,_=find_peaks(DOS_line,prominence=1)
        vbias_pks=vbias_list[pks]
        energies[x]=vbias_pks[np.abs(vbias_pks)<=vbias_max]
    return energies
        

if __name__=='__main__':
    # np.seterr(all='raise')
    parser=argparse.ArgumentParser()
    args=parse_arguments(parser)
    print(args)
    # nw.get_potential_type()

    st=time.time()
    
    x_range=np.linspace(args.x_min, args.x_max,args.x_num)
    y_range=np.linspace(args.y_min, args.y_max,args.y_num)
    inputs=[(args,x,y) for x in x_range for y in y_range]
    
    with MPIPoolExecutor() as executor:
        rs=list(tqdm(executor.map(wrapper,inputs),total=len(inputs)))
    # rs=list(map(wrapper,inputs))

    G_raw,TV_raw,kappa_raw,LDOS_raw,En_raw=zip(*rs)
    G=postprocess_G(G_raw)
    TV=postprocess_S(TV_raw)
    kappa=postprocess_S(kappa_raw)
    energies=postprocess_En(En_raw)
    LDOS=postprocess_LDOS(LDOS_raw)

    if args.energy and args.SE:
        energies=detect_peaks(LDOS)

    fn=filename(args)
    plot(fn=fn)
    savedata(fn=fn)
    
    print(time.time()-st)


