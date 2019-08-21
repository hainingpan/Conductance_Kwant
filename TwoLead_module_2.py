
# coding: utf-8

# In[1]:

import numpy as np
import kwant
import math
from math import pi
from cmath import sqrt
import numpy.linalg as LA

# =========== global variables ====================
s0 = np.array([[1.0, 0.0], [0.0, 1.0]]); sx = np.array([[0.0, 1.0], [1.0, 0.0]]); 
sy = np.array([[0.0, -1j], [1j, 0.0]]); sz = np.array([[1.0, 0.0], [0.0, -1.0]]);

t0 = np.array([[1.0, 0.0], [0.0, 1.0]]); tx = np.array([[0.0, 1.0], [1.0, 0.0]]);
ty = np.array([[0.0, -1j], [1j, 0.0]]); tz = np.array([[1.0, 0.0], [0.0, -1.0]]);

tzs0 = np.kron(tz,s0); t0s0 = np.kron(t0,s0); t0sx = np.kron(t0,sx);
txs0 = np.kron(tx,s0); tzsy = np.kron(tz,sy); t0sz = np.kron(t0,sz);


# In[327]:

def NSjunction(args_dict):
    t = args_dict['t'];
    alpha = args_dict['alpha'];
    Vz = args_dict['Vz'];
    Delta_0 = args_dict['Delta_0'];
    mu = args_dict['mu'];
    mu_lead = args_dict['mu_lead'];
    wireLength = args_dict['wireLength'];
    Nbarrier = args_dict['Nbarrier'];
    Ebarrier = args_dict['Ebarrier'];
    gamma = args_dict['gamma'];
    lamd=args_dict['lamd'];
    voltage=args_dict['voltage'];
    
    #========= set-up of NS-junction =========
    junction = kwant.Builder(); a=1; # Lattice constant
    lat = kwant.lattice.chain(a,norbs=4); 
    

    for x in range(wireLength):
        junction[ lat(x) ] = (2*t - mu)*tzs0 + Vz*t0sx + Delta_0*txs0 - 1j*gamma*t0s0;
    

    if args_dict['varymu']=='yes':
        mu1=args_dict['mu1']
        mu2=args_dict['mu2']
        uncoverLength=args_dict['uncoverLength']
        mus=np.linspace(mu1,mu2,wireLength)
        for x in range(wireLength):
            junction[ lat(x) ] = (2*t -mus[x])*tzs0 + Vz*t0sx + Delta_0*txs0 - 1j*gamma*t0s0
        for x in range(uncoverLength):
            junction[ lat(x) ] = (2*t -mus[x])*tzs0 + Vz*t0sx - 1j*gamma*t0s0

    if args_dict['SE'] == 'yes':
        SelfE=np.sign(voltage-Delta_0)*lamd*(voltage*t0s0+Delta_0*txs0)/sqrt(Delta_0**2-voltage**2+1e-9j)
        for x in range(wireLength):
            junction[ lat(x) ] = (2*t - mu)*tzs0 + Vz*t0sx+SelfE - 1j*gamma*t0s0;
	
    if args_dict['QD'] == 'yes':
        dotLength = args_dict['dotLength'];
        VD = args_dict['VD'];
        for x in range(0,dotLength):
           junction[ lat(x) ] = (2*t - mu + VD*np.cos(1.5*pi*(x)/dotLength) )*tzs0 + Vz*t0sx;
	
    if args_dict['QD2'] == 'yes': #Quantum Dot away from the lead.
            dotLength = args_dict['dotLength'];
            VD = args_dict['VD'];
            for x in range(0,dotLength):
                    junction[ lat(wireLength - x) ] = (2*t - mu + VD*np.cos(1.5*pi*(x)/dotLength) )*tzs0 + Vz*t0sx;

    
    for x in range(Nbarrier):
        junction[ lat(x) ] = (2*t - mu + Ebarrier)*tzs0 + Vz*t0sx;
        
    for x in range(wireLength - Nbarrier, wireLength):
        junction[ lat(x) ] = (2*t - mu + Ebarrier)*tzs0 + Vz*t0sx;
    
    for x in range( 1, wireLength ):
        junction[ lat(x-1), lat(x) ] = -t*tzs0 - 1j*alpha*tzsy;
    
    symLeft = kwant.TranslationalSymmetry([-a]);
    lead0 = kwant.Builder(symLeft, conservation_law=-tzs0);
    lead0[ lat(0) ] = (2*t - mu_lead)*tzs0 + Vz*t0sx;
    lead0[ lat(0), lat(1) ] = -t*tzs0 - 1j*alpha*tzsy;
    junction.attach_lead(lead0);
    
    symRight = kwant.TranslationalSymmetry([a]);
    lead1 = kwant.Builder(symRight, conservation_law=-tzs0);
    lead1[ lat(0) ] = (2*t - mu_lead)*tzs0 + Vz*t0sx;
    lead1[ lat(0), lat(1) ] = -t*tzs0 - 1j*alpha*tzsy;
    junction.attach_lead(lead1);
    
    junction = junction.finalized();
    return junction;

def conductance(args_dict):
    voltage = args_dict['voltage'];
    junction = NSjunction(args_dict);
    S_matrix = kwant.smatrix(junction, voltage, check_hermiticity=False);
    R_LL_ee = S_matrix.transmission((0,0),(0,0)); ## Equals S_matrix.submatrix((0,0),(0,0));
    R_LL_he = S_matrix.transmission((0,1),(0,0));
    R_RR_ee = S_matrix.transmission((1,0),(1,0)); ## Equals S_matrix.submatrix((0,0),(0,0));
    R_RR_he = S_matrix.transmission((1,1),(1,0));
    T_LR_ee = S_matrix.transmission((0,0),(1,0));
    T_LR_he = S_matrix.transmission((0,1),(1,0));
    T_RL_ee = S_matrix.transmission((1,0),(0,0)); ## Equals S_matrix.submatrix((1,0),(0,0));
    T_RL_he = S_matrix.transmission((1,1),(0,0));
    
    Nc = 2.0;
    Gr_LL = Nc - R_LL_ee + R_LL_he;
    Gr_RR = Nc - R_RR_ee + R_RR_he;
    GT_LR = T_LR_ee - T_LR_he;
    GT_RL = T_RL_ee - T_RL_he;

    return T_LR_ee,T_LR_he,T_RL_ee,T_RL_he,Gr_LL,Gr_RR,GT_LR,GT_RL;

def TV(args_dict):
    args_dict['voltage'] = 0.0; 
    junction = NSjunction(args_dict);
    
    S_matrix = kwant.smatrix(junction, args_dict['voltage'], check_hermiticity=False);
    R = S_matrix.submatrix(0,0); # "0" for The first lead index
    tv0 = LA.det(R);
    basis_wf = S_matrix.lead_info[0].wave_functions; # 'lead_info[i].wave_functions' contains the wavefunctions of the propagating modes in lead "i"
    
    normalize_dict = {0:0,1:0,2:3,3:3,4:0,5:0,6:3,7:3}
    phase_dict = {};
    
    for n in range(8):
        m = normalize_dict[n];
        phase_dict[n]= (-1)**m*basis_wf[m,n]/abs(basis_wf[m,n]); ##
        
    tv = tv0*np.conjugate(phase_dict[0]*phase_dict[1]*phase_dict[2]*phase_dict[3])*phase_dict[4]*phase_dict[5]*phase_dict[6]*phase_dict[7] ;
    
    return tv
    