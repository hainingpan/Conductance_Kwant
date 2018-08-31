import numpy as np
import kwant
import PauliMatrices as PM
from math import pi
from cmath import sqrt

def NSjunction(args_dict):
    a=args_dict['a'];   #lattice constant= 10a(nm)
    t = 25/(a*a);   #hopping integral
    alpha = args_dict['alpha_R']/(2*a);     #reduced SOC alpha=(alpha_R)/(2a)
    Vz = args_dict['Vz'];       #Zeeman field
    Delta_0 = args_dict['Delta_0'];     #Proximitized SC gap
    Delta_c=args_dict['Delta_c'];       #Coupling of two band in SC gap
    mu = args_dict['mu'];       #chemical potential with respect to band bottom
    mu_lead = args_dict['mu_lead'];     #chemical potential of lead
    wireLength = args_dict['wireLength'];       #Length of wire= $wireLength/100(um)
    Nbarrier = args_dict['Nbarrier'];   # number of barrier
    Ebarrier = args_dict['Ebarrier'];   
    gamma = args_dict['gamma'];     #dissipation
    lamd=args_dict['lamd'];     #SM and SC coupling strength
    voltage=args_dict['voltage'];       #bias voltage
    epsilon=args_dict['epsilon'];       #difference of bands
        
    junction=kwant.Builder();
    lat=kwant.lattice.chain(a);
    if args_dict['singleband']==1:
        for x in range(wireLength):
            junction[lat(x)]=(-mu+2*t)*PM.tzs0+Delta_0*PM.txs0+Vz*PM.t0sx-1j*gamma*PM.t0s0;
            
        for x in range(Nbarrier):
            junction[ lat(x) ] = (2*t - mu + Ebarrier)*PM.tzs0 + Vz*PM.t0sx;
        
        for x in range(1,wireLength):
            junction[lat(x-1),lat(x)]=-t*PM.tzs0-1j*alpha*PM.tzsy;
            
        symLeft=kwant.TranslationalSymmetry([-a]);
        lead=kwant.Builder(symLeft);
        lead[ lat(0) ] = (2*t - mu_lead)*PM.tzs0 + Vz*PM.t0sx;
        lead[ lat(0), lat(1) ] = -t*PM.tzs0 - 1j*alpha*PM.tzsy;
        junction.attach_lead(lead);
        junction=junction.finalized();
        return junction
    else:
        for x in range(wireLength):
            junction[lat(x)]=(-mu+2*t)*np.kron(np.array([[1,0],[0,0]]),PM.tzs0)+(epsilon-mu+2*t)*np.kron(np.array([[0,0],[0,1]]),PM.tzs0)+Delta_0*np.kron(PM.s0,PM.txs0)+Vz*np.kron(PM.s0,PM.t0sx)-1j*gamma*np.kron(PM.s0,PM.t0s0)+Delta_c*np.kron(PM.sx,PM.txs0);
        for x in range(1,wireLength):
            junction[lat(x-1),lat(x)]=-t*np.kron(PM.s0,PM.tzs0)-1j*alpha*np.kron(PM.s0,PM.tzsy);
        for x in range(Nbarrier):
            junction[ lat(x) ] = (2*t - mu + Ebarrier)*np.kron(PM.s0,PM.tzs0) + Vz*np.kron(PM.s0,PM.t0sx);
        symLeft=kwant.TranslationalSymmetry([-a]);
        lead=kwant.Builder(symLeft);
        lead[ lat(0) ] = (2*t - mu_lead)*np.kron(PM.s0,PM.tzs0) + Vz*np.kron(PM.s0,PM.t0sx);
        lead[ lat(0), lat(1) ] = -t*np.kron(PM.s0,PM.tzs0) - 1j*alpha*np.kron(PM.s0,PM.tzsy);
        junction.attach_lead(lead);
        junction=junction.finalized();
        return junction

def conductance(args_dict,junction):
    voltage=args_dict['voltage'];
    S_matrix = kwant.smatrix(junction, voltage, check_hermiticity=False);
    R = S_matrix.submatrix(0,0);
    if (args_dict['singleband']==1):
        G = 2.0;
        for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
            G = G - abs(R[i,j])**2 + abs(R[2+i,j])**2;
    else:
        G = 4.0;
        for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
#            G = G - abs(R[i,j])**2 + abs(R[2+i,j])**2-abs(R[4+i,4+j])**2+abs(R[6+i,4+j])**2;
#            G = G - abs(R[i,j])**2 + abs(R[2+i,j])**2-abs(R[4+i,4+j])**2+abs(R[6+i,4+j])**2-abs(R[i,4+j])**2-abs(R[4+i,j])**2+abs(R[6+i,j])**2+abs(R[2+i,4+j])**2;  
            G=G-abs(R[i,j])**2-abs(R[2+i,j])**2-abs(R[2+i,2+j])**2-abs(R[i,2+j])**2+abs(R[4+i,j])**2+abs(R[4+i+2,j])**2+abs(R[4+i,j+2])**2+abs(R[4+i+2,j+2])**2;               
#            G=G-abs(R[i,j])**2-abs(R[2+i,2+j])**2+abs(R[4+i,j])**2+abs(R[4+i+2,j+2])**2;               
            
    return G;
    

