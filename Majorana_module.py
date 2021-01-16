import numpy as np
import kwant
import PauliMatrices as PM
from math import pi
import numpy.linalg as LA
import matplotlib.pyplot as plt


def make_NS_junction(parameters):
    a=parameters['a']   #lattice constant= 10a(nm)
    t = 25/(a*a)   #hopping integral
    alpha = parameters['alpha_R']/(2.*a)     #reduced SOC alpha=(alpha_R)/(2a)
    vz = parameters['vz']       #Zeeman field
    delta0 = parameters['delta0']     #Proximitized SC gap
    mu = parameters['mu']       #chemical potential with respect to band bottom
    potPeak = parameters['potPeak']     #peak of smooth confinement
    muLead = parameters['muLead']     #chemical potential of lead
    wireLength = int(parameters['wireLength'])       #Length of wire= $wireLength/100(um)
    barrierNum = parameters['barrierNum']   # number of barrier
    barrierE = parameters['barrierE']
    couplingSCSM = parameters['couplingSCSM']   #SM-SC coupling strength
    dissipation = parameters['dissipation']     #dissipation
    vBias=parameters['vBias']       #bias voltage
    leadPos=parameters['leadPos']       #position of lead, 0: left 1: right
    potPeakPos=parameters['potPeakPos']   #position of the peak
    potSigma=parameters['potSigma']   #sigma(linewidth) in smooth potential or quatnum dot
    potPeakR=parameters['potPeakR']
    potPeakPosR=parameters['potPeakPosR']
    potSigmaR=parameters['potSigmaR']
    qdLength = int(parameters['qdLength'])    #length of quantum dot
    qdLengthR = int(parameters['qdLengthR'])   #lenght of right QD
    alpha_1=parameters['alpha']   #interpolate between inhomogeneous and disorder, (1-alpha)*V_imhom+alpha*V_imp, alpha =[0,1] is linear, otherwise unity.

    muVarList=parameters['muVarList']     #the spatial profile of disorder(V_impurity)
    N_muVar=parameters['N_muVar']    # number of muVar, used when length of muVarList is different from the number of sites
    vc = parameters['vc']   #The point where SC gap collapses. 0 for constant Delta (vc=infitity). The gap collapsing curve is delta_0*sqrt(1-(vz/vc)^2)
    randList=parameters['randList'] #the positive random list for only one in random {g,SC gap}. But does not support both.
    #N(1,gVar) for g
    #N(delta_0,deltaVar) for SC gap
    lat=kwant.lattice.chain(a,norbs=4)
    junction=kwant.Builder()

    #smooth confinement
    potential={
        0: lambda x: 0*x,
        'sin': lambda x: np.sin(x*pi/(0.1*wireLength))*potPeak,
        'sintheta': lambda x: potPeak*np.sin(x*pi/(wireLength/10))*(x<wireLength/10),
        'cos': lambda x: np.cos(3*x*pi/potSigma/2)*potPeak*(x<=potSigma),
        'cos2': lambda x: (np.cos(x*pi/potSigma/2)*potPeak*(x<=potSigma)-np.sin((x-potSigma)/20*pi)*(x>potSigma))*(x<=(potSigma+20)),
        'sin2': lambda x: np.sin(x*2*pi/wireLength)*potPeak,
        'sinabs': lambda x: np.abs(np.sin(x*2*pi/wireLength))*potPeak,
        'lorentz': lambda x: potPeak*1.0/(((x-potPeakPos*wireLength)*a)**2+0.5),
        'lorentzsigmoid': lambda x:  (potPeak*1.0/(((x-potPeakPos*wireLength)*a)**2+.5)+(4-mu)/2./(np.exp(-(x-0.5*wireLength)*a)+1)),
        'exp': lambda x: potPeak*(np.exp(-((x-potPeakPos)*a)**2/(2*potSigma**2))),
        'sigmoid': lambda x: potPeak*1/(np.exp((.5*wireLength-x)*a/potSigma)+1),
        'exp2': lambda x: potPeak*(np.exp(-((x-potPeakPos)*a)**2/(2*potSigma**2)))+potPeakR*(np.exp(-((x-potPeakPosR)*a)**2/(2*potSigmaR**2)))
    }

    muSet=mu-((not (alpha_1<=1 and alpha_1>=0))*1+(alpha_1<=1 and alpha_1>=0)*(1-alpha_1))*potential[parameters['potType']](np.arange(wireLength))
    if N_muVar!=1:
        muVarList=[muVarList.flatten()[int(N_muVar/wireLength*x)] for x in range(wireLength)]

    muSet=muSet-((not (alpha_1<=1 and alpha_1>=0))*1+(alpha_1<=1 and alpha_1>=0)*(alpha_1))*muVarList


    if parameters['deltaVar']==0:
        delta0=delta0*np.ones(wireLength)
    else:
        delta0=randList

    if vc!=0:
        if vz<vc:
            Delta=[x*np.sqrt(1-(vz/vc)**2) for x in delta0]
        else:
            Delta=np.zeros(wireLength)
    else:
        Delta=delta0

    if parameters['isSE']==0:
        scDelta=[x*PM.txs0 for x in Delta]
    else:
        scDelta=[-couplingSCSM*(vBias*PM.t0s0+x*PM.txs0)/np.sqrt(x**2-vBias**2-np.sign(vBias)*1e-9j)*(y!=0) for x,y in zip(Delta,delta0)]

    if parameters['isDissipationVar']!=0:
        dissipation=(vz/dissipation)**6/100

    if parameters['gVar']==0:
        vzSet=vz*np.ones(wireLength)
    else:
        vzSet=randList


    #Construct lattice  (multiband->scDelta& muSet not verified, the tau matrix should be replaced, gVar, deltaVar to be changed )
    for x in range(wireLength):
        junction[lat(x)]=(-muSet[x]+2*t)*PM.tzs0+scDelta[x]+vzSet[x]*PM.t0sx-1j*dissipation*PM.t0s0

    if parameters['isQD'] == 1:
        qdPeak = parameters['qdPeak']
        for x in range(qdLength):
            junction[ lat(x) ] = (2*t - mu + qdPeak*np.exp(-(x*a)**2/(qdLength*qdLength)) )*PM.tzs0 + vz*PM.t0sx - 1j*dissipation*PM.t0s0
#            junction[ lat(x) ] = (2*t - muSet[x])*PM.tzs0 + vz*PM.t0sx - 1j*dissipation*PM.t0s0
        qdPeakR = parameters['qdPeakR']
        for x in range(qdLengthR):
            junction[ lat(wireLength-x-1)] = (2*t - mu + qdPeakR*np.exp(-(x*a)**2/(qdLengthR*qdLengthR)) )*PM.tzs0 + vz*PM.t0sx - 1j*dissipation*PM.t0s0

    #Construct hopping
    junction[lat.neighbors()]=-t*PM.tzs0-1j*alpha*PM.tzsy


    #Construct lead and barrier
    if not (parameters['leadNum']==1 and parameters['leadPos']==1):   #exclude the situation of only right lead
        junction[(lat(x) for x in range(barrierNum))]=(2*t-mu+barrierE)*PM.tzs0 + vz*PM.t0sx
        symLeft=kwant.TranslationalSymmetry([-a])
        lead0=kwant.Builder(symLeft,conservation_law=-PM.tzs0)
        lead0[ lat(0) ] = (2*t - muLead)*PM.tzs0 + vz*PM.t0sx
        lead0[ lat(0), lat(1) ] = -t*PM.tzs0 - 1j*alpha*PM.tzsy
        junction.attach_lead(lead0)
    if not (parameters['leadNum']==1 and parameters['leadPos']==0):   #exclude the situation of only left lead
        junction[(lat(wireLength-x-1) for x in range(barrierNum))]=(2*t-mu+barrierE)*PM.tzs0 + vz*PM.t0sx
        symRight=kwant.TranslationalSymmetry([a])
        lead1=kwant.Builder(symRight,conservation_law=-PM.tzs0)
        lead1[ lat(0) ] = (2*t - muLead)*PM.tzs0 + vz*PM.t0sx
        lead1[ lat(0), lat(1) ] = -t*PM.tzs0 - 1j*alpha*PM.tzsy
        junction.attach_lead(lead1)

    #Finalize
    junction=junction.finalized()
    return junction

def conductance(parameters,junction):
    vBias=parameters['vBias']
    sMatrix = kwant.smatrix(junction, vBias, check_hermiticity=False)
    G=sMatrix.submatrix((0,0),(0,0)).shape[0]-sMatrix.transmission((0,0),(0,0))+sMatrix.transmission((0,1),(0,0))
    return G

def conductance_matrix(parameters,junction):
    vBias=parameters['vBias']
    sMatrix = kwant.smatrix(junction, vBias, check_hermiticity=False)
    #np.savetxt(str(vBias)+'.dat',sMatrix.data)
    # [[G_LL,G_LR],[G_RL,G_RR]]
    GLL=sMatrix.submatrix((0,0),(0,0)).shape[0]-sMatrix.transmission((0,0),(0,0))+sMatrix.transmission((0,1),(0,0))
    GRR=sMatrix.submatrix((1,0),(1,0)).shape[0]-sMatrix.transmission((1,0),(1,0))+sMatrix.transmission((1,1),(1,0))
    GLR=sMatrix.transmission((0,0),(1,0))-sMatrix.transmission((0,1),(1,0))
    GRL=sMatrix.transmission((1,0),(0,0))-sMatrix.transmission((1,1),(0,0))
    # return nparray or struct
    return GLL,GRR,GLR,GRL
'''
def topologicalQ(parameters,junction):
    vBias=parameters['vBias']
    sMatrix = kwant.smatrix(junction, vBias, check_hermiticity=False)
    if parameters['leadNum']==1:
        return np.abs(LA.det(sMatrix.data))
    if parameters['leadNum']==2:
        return np.abs(LA.det(sMatrix.submatrix(1,1)))
'''
def topologicalQ(parameters,junction):
    vBias=parameters['vBias']

    sMatrix = kwant.smatrix(junction, parameters['vBias'], check_hermiticity=False)
    R = sMatrix.submatrix(0,0)
    tv0 = LA.det(R)
    basis_wf = sMatrix.lead_info[0].wave_functions

    normalize_dict = {0:0,1:0,2:3,3:3,4:0,5:0,6:3,7:3}
    phase_dict = {}

    for n in range(8):
        m = normalize_dict[n]
        phase_dict[n]= (-1)**m*basis_wf[m,n]/abs(basis_wf[m,n])
    fixphase=np.conjugate(phase_dict[0]*phase_dict[1]*phase_dict[2]*phase_dict[3])* phase_dict[4]*phase_dict[5]*phase_dict[6]*phase_dict[7]
    # print('R-topo:')
    # print(R.round(3))
    # print('det='+str(tv0)+'  phase='+str(fixphase))
    tv = tv0*fixphase
    # print('tv='+str(tv))
    return tv

def getSMatrix(parameters,junction):
    vBias=parameters['vBias']
    sMatrix = kwant.smatrix(junction, parameters['vBias'], check_hermiticity=False)
    R = sMatrix.data
    basis_wf = sMatrix.lead_info[0].wave_functions

    normalize_dict = {0:0,1:0,2:3,3:3,4:0,5:0,6:3,7:3}
    phase_dict = {}

    for n in range(8):
        m = normalize_dict[n]
        phase_dict[n]= (-1)**m*basis_wf[m,n]/abs(basis_wf[m,n])

    fixphase=np.conjugate(phase_dict[0]*phase_dict[1]*phase_dict[2]*phase_dict[3])* phase_dict[4]*phase_dict[5]*phase_dict[6]*phase_dict[7]
    # print('R-sm: ')
    # R = R*fixphase
    # print(R[:4,:4].round(3))
    # print('det='+str(LA.det(R[:4,:4]))+'  phase='+str(fixphase))
    R = R.flatten().view(float)
    return R,fixphase

def TVmap(parameters,junction):
    vBias=parameters['vBias']   #TV is not suitable to define large value deviated from 0
    sMatrix = kwant.smatrix(junction, vBias, check_hermiticity=False)
    R = sMatrix.submatrix(0,0)
    tv0 = LA.det(R)
    basis_wf = sMatrix.lead_info[0].wave_functions
    normalize_dict = {0:0,1:0,2:3,3:3,4:0,5:0,6:3,7:3}
    phase_dict = {}
    for n in range(8):
        m = normalize_dict[n]
        phase_dict[n]= (-1)**m*basis_wf[m,n]/abs(basis_wf[m,n])

    tv = tv0*np.conjugate(phase_dict[0]*phase_dict[1]*phase_dict[2]*phase_dict[3])* phase_dict[4]    *phase_dict[5]*phase_dict[6]*phase_dict[7]
    return tv

def conductance_TV(parameters,junction):
    vBias=parameters['vBias']
    sMatrix = kwant.smatrix(junction, vBias, check_hermiticity=False)
    R = sMatrix.submatrix(0,0)
    G = 2.0
    for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
        G = G - abs(R[i,j])**2 + abs(R[2+i,j])**2

    tv0 = LA.det(R)
    basis_wf = sMatrix.lead_info[0].wave_functions
    normalize_dict = {0:0,1:0,2:3,3:3,4:0,5:0,6:3,7:3}
    phase_dict = {}
    for n in range(8):
        m = normalize_dict[n]
        phase_dict[n]= (-1)**m*basis_wf[m,n]/abs(basis_wf[m,n])

    tv = tv0*np.conjugate(phase_dict[0]*phase_dict[1]*phase_dict[2]*phase_dict[3])* phase_dict[4]    *phase_dict[5]*phase_dict[6]*phase_dict[7]

    return G,tv
