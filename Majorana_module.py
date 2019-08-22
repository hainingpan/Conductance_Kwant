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
    muVarlist=args_dict['muVarlist'];     #the spatial profile of disorder(V_impurity)
    Vc = args_dict['Vc'];   #The point where SC gap collapses. 0 for constant Delta (Vzc=infitity). The gap collapsing curve is delta_0*sqrt(1-(Vz/Vzc)^2) 
    randlist=args_dict['randlist']; #the positive random list for only one in random {g,SC gap,alpha_R,effective mass}. But does not support both. 
	#N(1,gVar) for g;
	#N(delta_0,DeltaVar) for SC gap
	#N(alpha_R,alpha_RVar) for alpha_R
	#N(1,massVar) for effective mass
    lat=kwant.lattice.chain(a,norbs=4);  
    junction=kwant.Builder();
     
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
    muset=muset-muVarlist;
    
    if args_dict['DeltaVar']==0:
        Delta_0=Delta_0*np.ones(wireLength);
    else:
        Delta_0=randlist;
        
    if Vc!=0:       
        if Vz<Vc:
            Delta=[x*np.sqrt(1-(Vz/Vc)**2) for x in Delta_0];
        else:
            Delta=np.zeros(wireLength);          
    else:
        Delta=Delta_0;           
         
    if args_dict['SE']==0:
        scDelta=[x*PM.txs0 for x in Delta];
    else:
        scDelta=[-gamma*(voltage*PM.t0s0+x*PM.txs0)/np.sqrt(x**2-voltage**2-np.sign(voltage)*1e-9j) for x in Delta];        
        
    if args_dict['GammaVar']!=0:
        Gamma=(Vz/Gamma)**6/100;
        
    if args_dict['gVar']==0:
        Vzlist=Vz*np.ones(wireLength);
    else:
        Vzlist=randlist;
		
    if args_dict['alpha_RVar']==0:
	    alphalist=alpha*np.ones(wireLength);
    else:
	    alphalist=randlist/(2.*a);        

#    if args_dict['massVar']==0:
#	    tlist=t*np.ones(wireLength);
#    else:
#	    tlist=t/randlist;   
        
    #Construct lattice  (multiband->scDelta& muset not verified, the tau matrix should be replaced, gVar, DeltaVar to be changed )
    for x in range(wireLength):
            junction[lat(x)]=(-muset[x]+2*t)*PM.tzs0+scDelta[x]+Vzlist[x]*PM.t0sx-1j*Gamma*PM.t0s0;
   
    if args_dict['QD'] == 1:
        VD = args_dict['VD'];
        for x in range(dotLength):
            junction[ lat(x) ] = (2*t - mu + VD*np.exp(-x*x/(dotLength*dotLength)) )*PM.tzs0 + Vz*PM.t0sx - 1j*Gamma*PM.t0s0;
    #Construct hopping
    for x in range(1,wireLength):
            junction[lat(x-1),lat(x)]=-t*PM.tzs0-1j*alphalist[x]*PM.tzsy;
   
    #Construct barrier
#    for x in range(Nbarrier):
#           barrierindex=int(-0.5+((leadpos==0)-(leadpos==1))*(x+1))%wireLength;
#            junction[ lat(barrierindex) ] = (2*t- mu + Ebarrier)*PM.tzs0 + Vz*PM.t0sx;
    
    #Construct lead and barrier
    if not (args_dict['leadnum']==1 and args_dict['leadpos']==1):   #exclude the situation of only right lead
        junction[(lat(x) for x in range(Nbarrier))]=(2*t-mu+Ebarrier)*PM.tzs0 + Vz*PM.t0sx;
        symLeft=kwant.TranslationalSymmetry([-a]);
        lead0=kwant.Builder(symLeft,conservation_law=-PM.tzs0);
        lead0[ lat(0) ] = (2*t - mu_lead)*PM.tzs0 + Vz*PM.t0sx;
        lead0[ lat(0), lat(1) ] = -t*PM.tzs0 - 1j*alpha*PM.tzsy;
        junction.attach_lead(lead0);
    if not (args_dict['leadnum']==1 and args_dict['leadpos']==0):   #exclude the situation of only left lead
        junction[(lat(wireLength-x-1) for x in range(Nbarrier))]=(2*t-mu+Ebarrier)*PM.tzs0 + Vz*PM.t0sx;
        symRight=kwant.TranslationalSymmetry([a]);
        lead1=kwant.Builder(symRight,conservation_law=-PM.tzs0);
        lead1[ lat(0) ] = (2*t - mu_lead)*PM.tzs0 + Vz*PM.t0sx;
        lead1[ lat(0), lat(1) ] = -t*PM.tzs0 - 1j*alpha*PM.tzsy;
        junction.attach_lead(lead1);
    
    #Finalize
    junction=junction.finalized();
    return junction
 
def conductance(args_dict,junction):
    voltage=args_dict['voltage'];
    S_matrix = kwant.smatrix(junction, voltage, check_hermiticity=False);
    G=S_matrix.submatrix((0,0),(0,0)).shape[0]-S_matrix.transmission((0,0),(0,0))+S_matrix.transmission((0,1),(0,0)) 
    return G;
    
def conductance_matrix(args_dict,junction):
    voltage=args_dict['voltage'];
    S_matrix = kwant.smatrix(junction, voltage, check_hermiticity=False);
    # [[G_LL,G_LR],[G_RL,G_RR]]
    GLL=S_matrix.submatrix((0,0),(0,0)).shape[0]-S_matrix.transmission((0,0),(0,0))+S_matrix.transmission((0,1),(0,0)) 
    GRR=S_matrix.submatrix((1,0),(1,0)).shape[0]-S_matrix.transmission((1,0),(1,0))+S_matrix.transmission((1,1),(1,0)) 
    GLR=S_matrix.transmission((0,0),(1,0))-S_matrix.transmission((0,1),(1,0))
    GRL=S_matrix.transmission((1,0),(0,0))-S_matrix.transmission((1,1),(0,0))
    # return nparray or struct
    return GLL,GRR,GLR,GRL

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
    G = 2.0;
    for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
        G = G - abs(R[i,j])**2 + abs(R[2+i,j])**2;

    tv0 = LA.det(R);
    basis_wf = S_matrix.lead_info[0].wave_functions;    
    normalize_dict = {0:0,1:0,2:3,3:3,4:0,5:0,6:3,7:3}
    phase_dict = {};    
    for n in range(8):
        m = normalize_dict[n];
        phase_dict[n]= (-1)**m*basis_wf[m,n]/abs(basis_wf[m,n]);
        
    tv = tv0*np.conjugate(phase_dict[0]*phase_dict[1]*phase_dict[2]*phase_dict[3])* phase_dict[4]    *phase_dict[5]*phase_dict[6]*phase_dict[7] ;
    
    return G,tv
