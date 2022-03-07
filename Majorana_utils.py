import numpy as np
import kwant
import tinyarray
import time
import matplotlib.pyplot as plt
from itertools import repeat

s0 = tinyarray.array([[1, 0], [0, 1]])
sx = tinyarray.array([[0, 1], [1, 0]])
sy = tinyarray.array([[0, -1j], [1j, 0]])
sz = tinyarray.array([[1, 0], [0, -1]])

t0 = tinyarray.array([[1, 0], [0, 1]])
tx = tinyarray.array([[0, 1], [1, 0]])
ty = tinyarray.array([[0, -1j], [1j, 0]])
tz = tinyarray.array([[1, 0], [0, -1]])

t0s0 = np.kron(s0,s0)
t0sx = np.kron(s0,sx)
t0sy = np.kron(s0,sy)
t0sz = np.kron(s0,sz)

txs0 = np.kron(sx,s0)
txsx = np.kron(sx,sx)
txsy = np.kron(sx,sy)
txsz = np.kron(sx,sz)

tys0 = np.kron(sy,s0)
tysx = np.kron(sy,sx)
tysy = np.kron(sy,sy)
tysz = np.kron(sy,sz)

tzs0 = np.kron(sz,s0)
tzsx = np.kron(sz,sx)
tzsy = np.kron(sz,sy)
tzsz = np.kron(sz,sz)


_m_e=9.1093837015e-31 # kg
_hbar=1.0545718e-34 # J.s
_e=1.60217662e-19 # C
eps=1e-10
class Nanowire:
    def __init__(self,args):
        '''
        Initialize Nanowire parameters
    
        Parameters
        ----------
        args : argsparse.ArgumentParser
                The input arguments for the nanowire
        '''
        self.args=args
        self.t=1e3/_e*_hbar**2/(2*self.args.mass*_m_e*(self.args.a*1e-9)**2)
        # self.t=25
        self.alpha_R=self.args.alpha/(2*self.args.a*10)*1000
        self.wire_num=self.args.L*1000/self.args.a
        assert abs(self.wire_num-round(self.args.L*1000/self.args.a))<eps, 'The wire length ({} um) and the lattice constant ({} nm) are not commensurate.'.format(self.args.L,self.args.a)
        self.wire_num=int(self.wire_num)
        self.QD_num=self.args.QD_L*1000/self.args.a
        assert abs(self.QD_num-round(self.QD_num))<eps,'The length of the quantum dot on the left ({} um) and the lattice constant ({} nm) are not commensurate.'.format(self.args.QD_L,self.args.a)
        self.QD_num=int(self.QD_num)
        self.QD_num_R=self.args.QD_L_R*1000/self.args.a
        assert abs(self.QD_num_R-round(self.QD_num_R))<eps, 'The length of the quantum dot on the right ({} um) and the lattice constant ({} nm) are not commensurate.'.format(self.args.QD_L,self.args.a)
        self.QD_num_R=int(self.QD_num_R)
        assert self.args.lead_num==len(self.args.lead_pos), 'The number of leads ({}) is not equal to the position information ({})'.format(args.lead_num,args.lead_pos)

        if len(self.args.muVar_fn)>0:
            print('Use guassian disorder file:{}'.format(self.args.muVar_fn))
            try:
                self.muVar_list=np.loadtxt(self.args.muVar_fn)
            except:
                raise ValueError('Error in reading disorder file: {}.'.format(self.args.muVar_fn))
            assert len(self.muVar_list)==self.wire_num, 'The length of muVar_list ({}) is not equal to the number of the unit cell of wire ({}).'.format(len(self.muVar_list),self.wire_num)
        else:  
            rng_muVar=np.random.default_rng(self.args.muVar_seed)          
            self.muVar_list=rng_muVar.normal(size=self.wire_num)*self.args.muVar

        if len(self.args.random_fn)>0:
            print('Use random file:'.format(self.args.random_fn))
            try:
                self.random_list=np.loadtxt(self.args.random_fn)
            except:
                raise ValueError('Error in the disorder file:{}'.format(self.args.random_fn))
            assert len(self.random_list)==self.wire_num, 'The length of random_list ({}) is not equal to the number of the unit cell of wire ({}).'.format(len(self.random_list),self.wire_num)
        else:
            assert self.args.gVar>=0, 'gVar ({}) should be non-negative'.format(self.args.gVar)
            assert self.args.DeltaVar>=0, 'DeltaVar ({}) should be non-negative'.format(self.args.DeltaVar)
            assert self.args.coupling_SC_SM_Var>=0, 'coupling_SC_SM_Var ({}) should be non-negative'.format(self.args.coupling_SC_SM_Var)

            count_nonzero=0
            count_nonzero+=0 if self.args.gVar==0 else 1
            count_nonzero+=0 if self.args.DeltaVar==0 else 1
            count_nonzero+=0 if self.args.coupling_SC_SM_Var==0 else 1
            # almost one of g, Delta, coupling_SC_SM is nonzero
            assert count_nonzero<=1, 'More than one parameter is random (gVar={},DeltaVar={},coupling_SC_SM_Var={}).'.format(self.args.gVar,self.args.DeltaVar,self.args.coupling_SC_SM_Var)
            rng_rand=np.random.default_rng(self.args.random_seed)
            self.random_list=rng_rand.normal(size=self.wire_num)*max(self.args.gVar,self.args.DeltaVar,self.args.coupling_SC_SM_Var)
            while min(self.random_list)<0:
                self.random_list=rng_rand.normal(size=self.wire_num)*max(self.args.gVar,self.args.DeltaVar,self.args.coupling_SC_SM_Var)

        assert self.args.x!=self.args.y, 'x-axis {} and y-axis {} are the same.'.format(self.args.x,self.args.y)
        parameter_knob=set(['mu','alpha_R','Delta0','mu_lead','barrier_E','dissipation','QD_peak','QD_L','QD_peak_R','QD_L_R','coupling_SC_SM','Vzc','pot_peak_pos','pot_sigma','pot_peak','pot_peak_R','pot_peak_pos_R','pot_sigma_R','Vz','V_bias'])
        assert self.args.x in parameter_knob, 'x-axis does not support {}'.format(self.args.x)
        assert self.args.y in parameter_knob, 'y-axis does not support {}'.format(self.args.y)

        self._potential={
        '0': lambda x: 0*x,
        'cos': lambda x: np.cos(3*x*np.pi/self.args.pot_sigma/2)*self.args.pot_peak*(x<=self.args.pot_sigma),
        'exp': lambda x: self.args.pot_peak*(np.exp(-((x-self.args.pot_peak_pos))**2/(2*self.args.pot_sigma**2))),
        'exp2': lambda x: self.args.pot_peak*(np.exp(-((x-self.args.pot_peak_pos))**2/(2*self.args.pot_sigma**2)))+self.args.pot_peak_R*(np.exp(-((x-self.args.pot_peak_pos_R))**2/(2*self.args.pot_sigma_R**2)))        
        }
        assert self.args.pot_type in self._potential, 'Potential type ({}) is not defined.'.format(self.args.pot_type)

    def get_potential_type(self):
        y=self._potential[self.args.pot_type](np.linspace(0,self.args.L,self.wire_num))
        fig,ax=plt.subplots()
        ax.plot(np.arange(self.wire_num)*self.args.a/1000,y)
        ax.set_title(self.args.pot_type)
        ax.set_xlabel('L($\mu$m)')
        ax.set_ylabel('V(meV)')
        fig.savefig('{}.png'.format(self.args.pot_type))

    def get_hamiltonian_bare(self):
        self.lat=kwant.lattice.chain(self.args.a,norbs=4)
        self.hamiltonian_bare=kwant.Builder()

        self._mu_list()
        self._SC_Delta_list()      
        self._Vz_list()
        for x in range(self.wire_num):
            self.hamiltonian_bare[self.lat(x)]=(-self.mu_list[x]+2*self.t)*tzs0+self.SC_Delta_list[x]+self.Vz_list[x]*t0sx-1j*self.args.dissipation*t0s0
        self._QD()
        self.hamiltonian_bare[self.lat.neighbors()]=-self.t*tzs0-1j*self.alpha_R*tzsy
        return self.hamiltonian_bare
        
    def _mu_list(self):
        '''disordered and inhomogeneous potential'''
        self.mu_list=self.args.mu-self._potential[self.args.pot_type](np.linspace(0,self.args.L,self.wire_num))
        self.mu_list=self.mu_list-self.muVar_list

    def _SC_Delta_list(self):
        ''' random disorder in SC'''
        self.Delta0_list=self.args.Delta0*np.ones(self.wire_num) if self.args.DeltaVar==0 else self.random_list
        self.coupling_SC_SM_list=self.args.coupling_SC_SM*np.ones(self.wire_num) if self.args.coupling_SC_SM_Var==0 else self.random_list
        self.Delta_list=np.zeros(self.wire_num) if self.args.Vz>self.args.Vzc else self.Delta0_list*np.sqrt(1-(self.args.Vz/self.args.Vzc)**2)
        self.SC_Delta_list=[-coupling_SC_SM*(self.args.V_bias*t0s0+Delta*txs0)/np.sqrt(Delta**2-self.args.V_bias**2-np.sign(self.args.V_bias)*1e-9j)*(Delta0>0) for Delta, Delta0, coupling_SC_SM  in zip(self.Delta_list,self.Delta0_list,self.coupling_SC_SM_list)] if self.args.SE else [Delta*txs0 for Delta in self.Delta_list]
    
    def _Vz_list(self):
        '''random disorder in g factor'''
        self.Vz_list=self.args.Vz*np.ones(self.wire_num) if self.args.gVar==0 else self.random_list

    def _QD(self):
        ''' quantum dot '''
        if self.args.QD:
            for x in range(self.QD_num):
                self.hamiltonian_bare[self.lat(x)]=(2*self.t-self.args.mu+self.args.QD_peak*np.exp(-(x*self.args.a)**2/self.args.QD_L**2))*tzs0+self.Vz_list[x]*t0sx-1j*self.args.dissipation*t0s0
            for x in range(self.QD_num_R):
                self.hamiltonian_bare[self.lat(self.wire_num-x-1)]=(2*self.t-self.args.mu+self.args.QD_peak_R*np.exp(-(x*self.args.a)**2/self.args.QD_L_R**2))*tzs0+self.Vz_list[x]*t0sx-1j*self.args.dissipation*t0s0
        
    def _lead(self,junction,lead_pos):
        if self.args.barrier_relative is not None:
            self.args.barrier_E=self.args.mu+self.args.barrier_relative
        if lead_pos=='L':
            junction[(self.lat(x) for x in range(self.args.barrier_num))]=(2*self.t-self.args.mu+self.args.barrier_E)*tzs0+ self.args.Vz*t0sx
            sym_L=kwant.TranslationalSymmetry([-self.args.a])
            lead_L=kwant.Builder(sym_L,conservation_law=-tzs0)
            lead_L[self.lat(0)]=(2*self.t-self.args.mu_lead)*tzs0+self.args.Vz*t0sx
            lead_L[self.lat(0),self.lat(1)]=-self.t*tzs0-1j*self.alpha_R*tzsy
            junction.attach_lead(lead_L)
        elif lead_pos=='R':
            junction[(self.lat(self.wire_num-x-1) for x in range(self.args.barrier_num))]=(2*self.t-self.args.mu+self.args.barrier_E)*tzs0+ self.args.Vz*t0sx
            sym_R=kwant.TranslationalSymmetry([self.args.a])
            lead_R=kwant.Builder(sym_R,conservation_law=-tzs0)
            lead_R[self.lat(0)]=(2*self.t-self.args.mu_lead)*tzs0+self.args.Vz*t0sx
            lead_R[self.lat(0),self.lat(1)]=-self.t*tzs0-1j*self.alpha_R*tzsy
            junction.attach_lead(lead_R)
        return junction

    def get_hamiltonian_lead(self,lead_pos=None):
        if not hasattr(self, 'hamitonian_bare'):
            self.get_hamiltonian_bare()
        if self.args.lead_num==1:
            junction=self._lead(self.hamiltonian_bare,lead_pos)            
        elif self.args.lead_num==2:
            junction=self._lead(self.hamiltonian_bare,'R')
            junction=self._lead(self.hamiltonian_bare,'L')
        return junction

    def _Green_function(self,ham,delta=1e-3):
        '''G= -1/pi* imag( w-H)^(-1)
        '''
        GF=-1/np.pi*np.imag(np.linalg.inv((self.args.V_bias+1j*delta)*np.eye(ham.shape[0])-ham))
        return np.diag(GF).reshape((-1,4)).sum(axis=1)

    def LDOS(self,x,y):
        '''full with x, y (has to be with energy/V_bias),  L
        '''
        setattr(self.args, self.args.x,x)
        setattr(self.args, self.args.y,y)
        if not hasattr(self, 'hamitonian_bare'):
            self.get_hamiltonian_bare()
        ham=self.hamiltonian_bare.finalized()
        return self._Green_function(ham.hamiltonian_submatrix())

    def conductance(self,x,y):
        setattr(self.args, self.args.x,x)
        setattr(self.args, self.args.y,y)
        G={}
        if self.args.lead_num==1:
            for lead_pos in self.args.lead_pos:
                hamiltonian_lead=self.get_hamiltonian_lead(lead_pos).finalized()
                s_matrix=kwant.smatrix(hamiltonian_lead,self.args.V_bias,check_hermiticity=False)
                G[lead_pos]=s_matrix.submatrix((0,0),(0,0)).shape[0]-s_matrix.transmission((0,0),(0,0))+s_matrix.transmission((0,1),(0,0))
            S,TVL,TVR=repeat(None,3)
        elif self.args.lead_num==2:
            hamiltonian_lead=self.get_hamiltonian_lead().finalized()
            s_matrix=kwant.smatrix(hamiltonian_lead,self.args.V_bias,check_hermiticity=False)
            G['LL']=s_matrix.submatrix((0,0),(0,0)).shape[0]-s_matrix.transmission((0,0),(0,0))+s_matrix.transmission((0,1),(0,0))
            G['RR']=s_matrix.submatrix((1,0),(1,0)).shape[0]-s_matrix.transmission((1,0),(1,0))+s_matrix.transmission((1,1),(1,0))
            G['LR']=s_matrix.transmission((0,0),(1,0))-s_matrix.transmission((0,1),(1,0))
            G['RL']=s_matrix.transmission((1,0),(0,0))-s_matrix.transmission((1,1),(0,0))
            if abs(self.args.V_bias)<eps:
                S=s_matrix.data
                basis_wf = s_matrix.lead_info[0].wave_functions
                normalize=[0,0,3,3,0,0,3,3]
                phase = np.array([(-1)**m*basis_wf[m,n]/abs(basis_wf[m,n]) for n,m in enumerate(normalize)])
                fixphase=np.conj(np.prod(phase[:4]))*np.prod(phase[4:])
                TVL,TVR=np.linalg.det(S[:4,:4]),np.linalg.det(S[4:,4:])
                assert (np.imag(TVL)<eps and np.imag(TVR)<eps),'TVL and TVR are not real with imag=({:e},{:e})'.format(np.imag(TVL),np.imag(TVR))
                TVL=np.real(fixphase*TVL)
                TVR=np.real(fixphase*TVR)
            else:
                S,TVL,TVR=repeat(None,3)
        return G,S,TVL,TVR

        



        
        



