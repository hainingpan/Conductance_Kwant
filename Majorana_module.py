import numpy as np
import kwant
import PauliMatrices as PM
from math import pi
import numpy.linalg as LA
import matplotlib.pyplot as plt


def NSjunction(args_dict):
    a=args_dict['a'];   #lattice constant= 10a(nm)
    t = 25/(a*a);   #hopping integral
    alpha = args_dict['alpha_R']/(2.*a);     #reduced SOC alpha=(alpha_R)/(2a)
    Vz = args_dict['Vz'];       #Zeeman field
    Delta_0 = args_dict['Delta_0'];     #Proximitized SC gap
    Delta_c=args_dict['Delta_c'];       #Coupling of two band in SC gap
    mu = args_dict['mu'];       #chemical potential with respect to band bottom
    mumax = args_dict['mumax'];     #peak of smooth confinement
    mu_lead = args_dict['mu_lead'];     #chemical potential of lead
    wireLength = int(args_dict['wireLength']);       #Length of wire= $wireLength/100(um)
    Nbarrier = args_dict['Nbarrier'];   # number of barrier
    Ebarrier = args_dict['Ebarrier'];  
    gamma = args_dict['gamma'];   #SM-SC coupling strength
    Gamma = args_dict['Gamma'];     #dissipation
    voltage=args_dict['voltage'];       #bias voltage
    epsilon=args_dict['epsilon'];       #difference of bands
    leadpos=args_dict['leadpos'];       #position of lead, 0: left; 1: right
    peakpos=args_dict['peakpos'];   #position of the peak
    sigma=args_dict['sigma'];   #sigma(linewidth) in smooth potential or quatnum dot
    dotLength = int(args_dict['dotLength']);    #length of quantum dot
    vimplist=args_dict['vimplist'];     #the spatial profile of disorder(V_impurity)
    Vc = args_dict['Vc'];   #The point where SC gap collapses. 0 for constant Delta (Vzc=infitity). The gap collapsing curve is delta_0*sqrt(1-(Vz/Vzc)^2) 
    
    junction=kwant.Builder();
    lat=kwant.lattice.chain(a);  
    #smooth confinement
    potential={
        0: lambda x: mu*x**0,
        'sin': lambda x: np.sin(x*pi/(0.1*wireLength))*mumax+mu,
        'sintheta': lambda x: mumax*np.sin(x*pi/(wireLength/10))*(x<wireLength/10)+mu,
        'cos': lambda x: np.cos(x*pi/wireLength)*mumax+mu,
        'sin2': lambda x: np.sin(x*2*pi/wireLength)*mumax+mu,
        'sinabs': lambda x: np.abs(np.sin(x*2*pi/wireLength))*mumax+mu,
        'lorentz': lambda x: mumax*1.0/(((x-peakpos*wireLength)*a)**2+0.5)+mu,
        'lorentzsigmoid': lambda x:  (mumax*1.0/(((x-peakpos*wireLength)*a)**2+.5)+(4-mu)/2./(np.exp(-(x-0.5*wireLength)*a)+1))+mu, 
        'exp': lambda x: mumax*(np.exp(-((x-peakpos*wireLength)*a)**2/(2*sigma**2)))+mu,
        'sigmoid': lambda x: mu+mumax*1/(np.exp((.5*wireLength-x)*a/sigma)+1)
    }
    muset=potential[args_dict['smoothpot']](np.arange(wireLength));     
    muset=muset-vimplist;

    if Vc!=0:
        Delta=Delta_0*np.sqrt(1-(Vz/Vc)^2);
    else:
        Delta=Delta_0;            
         
    if args_dict['SE']==0:
        scDelta=Delta;
    else:
        scDelta=-gamma*(voltage*PM.t0s0+Delta*PM.txs0)/np.sqrt(Delta**2-voltage**2+np.sign(voltage)*1e-9j);        
        
    #Construct lattice  (multiband->scDelta& muset not verified)
    if args_dict['multiband']==0:
        for x in range(wireLength):
            junction[lat(x)]=(-muset[x]+2*t)*PM.tzs0+scDelta*PM.txs0+Vz*PM.t0sx-1j*Gamma*PM.t0s0;
    else:
        for x in range(wireLength):
            junction[lat(x)]=(-muset[x]+2*t)*np.kron(np.array([[1,0],[0,0]]),PM.tzs0)+(epsilon-muset[x]+2*t)*np.kron(np.array([[0,0],[0,1]]),PM.tzs0)+scDelta*np.kron(PM.s0,PM.txs0)+Vz*np.kron(PM.s0,PM.t0sx)-1j*Gamma*np.kron(PM.s0,PM.t0s0)+Delta_c*np.kron(PM.sx,PM.txs0);
   
    if args_dict['QD'] == 1:
        VD = args_dict['VD'];
        for x in range(dotLength):
            junction[ lat(x) ] = (2*t - mu + VD*np.exp(-x*x/(dotLength*dotLength)) )*PM.tzs0 + Vz*PM.t0sx - 1j*Gamma*PM.t0s0;
    #Construct hopping
    if args_dict['multiband']==0:
        for x in range(1,wireLength):
            junction[lat(x-1),lat(x)]=-t*PM.tzs0-1j*alpha*PM.tzsy;
    else:
        for x in range(1,wireLength):
            junction[lat(x-1),lat(x)]=-t*np.kron(PM.s0,PM.tzs0)-1j*alpha*np.kron(PM.s0,PM.tzsy);
    #Construct barrier
    
    if args_dict['multiband']==0:
        for x in range(Nbarrier):
            barrierindex=int(-0.5+((leadpos==0)-(leadpos==1))*(x+1))%wireLength;
            junction[ lat(barrierindex) ] = (2*t - mu + Ebarrier)*PM.tzs0 + Vz*PM.t0sx;
    else:
        for x in range(Nbarrier):
            barrierindex=int(-0.5+((leadpos==0)-(leadpos==1))*(x+1))%wireLength;
            junction[ lat(barrierindex) ] = (2*t - mu + Ebarrier)*np.kron(PM.s0,PM.tzs0) + Vz*np.kron(PM.s0,PM.t0sx);
    #Consruct lead
    symLeft=kwant.TranslationalSymmetry([-a]);
    symRight=kwant.TranslationalSymmetry([a]);
    if leadpos==0:
        lead=kwant.Builder(symLeft);
    else:
        lead=kwant.Builder(symRight);
        
    if args_dict['multiband']==0:
        lead[ lat(0) ] = (2*t - mu_lead)*PM.tzs0 + Vz*PM.t0sx;
        lead[ lat(0), lat(1) ] = -t*PM.tzs0 - 1j*alpha*PM.tzsy;
    else:
        lead[ lat(0) ] = (2*t - mu_lead)*np.kron(PM.s0,PM.tzs0) + Vz*np.kron(PM.s0,PM.t0sx);
        lead[ lat(0), lat(1) ] = -t*np.kron(PM.s0,PM.tzs0) - 1j*alpha*np.kron(PM.s0,PM.tzsy);       
    #Attach lead
    junction.attach_lead(lead);
    #Finalize
    junction=junction.finalized();
    return junction
 
def conductance(args_dict,junction):
    voltage=args_dict['voltage'];
    S_matrix = kwant.smatrix(junction, voltage, check_hermiticity=False);
    R = S_matrix.submatrix(0,0);
    if (args_dict['multiband']==0):
        G = 2.0;
        for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
            G = G - abs(R[i,j])**2 + abs(R[2+i,j])**2;
    else:
        G = 4.0;
        for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
            G=G-abs(R[i,j])**2-abs(R[2+i,j])**2-abs(R[2+i,2+j])**2-abs(R[i,2+j])**2+abs(R[4+i,j])**2+abs(R[4+i+2,j])**2+abs(R[4+i,j+2])**2+abs(R[4+i+2,j+2])**2;          
    return G;
    
def TV(args_dict):
    args_dict['voltage'] = 0.0; 
    junction = NSjunction(args_dict);
    
    S_matrix = kwant.smatrix(junction, args_dict['voltage'], check_hermiticity=False);
    R = S_matrix.submatrix(0,0);
    tv0 = LA.det(R);
    basis_wf = S_matrix.lead_info[0].wave_functions;
    
    normalize_dict = {0:0,1:0,2:3,3:3,4:0,5:0,6:3,7:3}
    phase_dict = {};
    
    for n in range(8):
        m = normalize_dict[n];
        phase_dict[n]= (-1)**m*basis_wf[m,n]/abs(basis_wf[m,n]);
        
    tv = tv0*np.conjugate(phase_dict[0]*phase_dict[1]*phase_dict[2]*phase_dict[3])* phase_dict[4]    *phase_dict[5]*phase_dict[6]*phase_dict[7] ;
    
    return tv
    
def TVmap(args_dict,junction):
    voltage=args_dict['voltage'];   
    S_matrix = kwant.smatrix(junction, voltage, check_hermiticity=False);
    R = S_matrix.submatrix(0,0);
    tv0 = LA.det(R);
    basis_wf = S_matrix.lead_info[0].wave_functions;    
    normalize_dict = {0:0,1:0,2:3,3:3,4:0,5:0,6:3,7:3}
    phase_dict = {};    
    for n in range(8):
        m = normalize_dict[n];
        phase_dict[n]= (-1)**m*basis_wf[m,n]/abs(basis_wf[m,n]);
        
    tv = tv0*np.conjugate(phase_dict[0]*phase_dict[1]*phase_dict[2]*phase_dict[3])* phase_dict[4]    *phase_dict[5]*phase_dict[6]*phase_dict[7] ;
    return tv
    
def ConductanceAndTV(args_dict,junction):
    voltage=args_dict['voltage'];
    S_matrix = kwant.smatrix(junction, voltage, check_hermiticity=False);
    R = S_matrix.submatrix(0,0);
    if (args_dict['multiband']==0):
        G = 2.0;
        for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
            G = G - abs(R[i,j])**2 + abs(R[2+i,j])**2;
    else:
        G = 4.0;
        for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
            G=G-abs(R[i,j])**2-abs(R[2+i,j])**2-abs(R[2+i,2+j])**2-abs(R[i,2+j])**2+abs(R[4+i,j])**2+abs(R[4+i+2,j])**2+abs(R[4+i,j+2])**2+abs(R[4+i+2,j+2])**2;          
    

    tv0 = LA.det(R);
    basis_wf = S_matrix.lead_info[0].wave_functions;    
    normalize_dict = {0:0,1:0,2:3,3:3,4:0,5:0,6:3,7:3}
    phase_dict = {};    
    for n in range(8):
        m = normalize_dict[n];
        phase_dict[n]= (-1)**m*basis_wf[m,n]/abs(basis_wf[m,n]);
        
    tv = tv0*np.conjugate(phase_dict[0]*phase_dict[1]*phase_dict[2]*phase_dict[3])* phase_dict[4]    *phase_dict[5]*phase_dict[6]*phase_dict[7] ;
    
    return G,tv
