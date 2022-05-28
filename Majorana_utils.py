import numpy as np
import kwant
import tinyarray
import time
import matplotlib.pyplot as plt
from itertools import repeat
from scipy.sparse.linalg import eigsh

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
_eps=1e-10
_TV_eps=1e-5
_kappa_eps=1e-5
class Nanowire:
    '''
    The class to represent the 1D nanowire
    '''
    def __init__(self,args):
        '''
        Initialize Nanowire parameters.
    
        Parameters
        ----------
        args : argsparse.ArgumentParser
                The input arguments for the nanowire.
        '''
        self.args=args
        self.t=1e3/_e*_hbar**2/(2*self.args.mass*_m_e*(self.args.a*1e-9)**2)
        # self.t=25
        self.alpha_R=self.args.alpha/(2*self.args.a*10)*1000
        self.wire_num=self.args.L*1000/self.args.a
        assert abs(self.wire_num-round(self.args.L*1000/self.args.a))<_eps, 'The wire length ({} um) and the lattice constant ({} nm) are not commensurate.'.format(self.args.L,self.args.a)
        self.wire_num=int(self.wire_num)
        self.QD_num=self.args.QD_L*1000/self.args.a
        assert abs(self.QD_num-round(self.QD_num))<_eps,'The length of the quantum dot on the left ({} um) and the lattice constant ({} nm) are not commensurate.'.format(self.args.QD_L,self.args.a)
        self.QD_num=int(self.QD_num)
        self.QD_num_R=self.args.QD_L_R*1000/self.args.a
        assert abs(self.QD_num_R-round(self.QD_num_R))<_eps, 'The length of the quantum dot on the right ({} um) and the lattice constant ({} nm) are not commensurate.'.format(self.args.QD_L,self.args.a)
        self.QD_num_R=int(self.QD_num_R)
        assert np.abs(self.args.y_max-self.args.y_min)/self.args.y_num>_eps, 'The number of points on y axis ({}) is too large.'.format(self.args.y_num)
        # assert self.args.lead_num==len(self.args.lead_pos), 'The number of leads ({}) is not equal to the position information ({})'.format(args.lead_num,args.lead_pos)

        if len(self.args.muVar_fn)>0:
            # print('Use guassian disorder file:{}'.format(self.args.muVar_fn))
            try:
                self.muVar_list=np.loadtxt(self.args.muVar_fn)
            except:
                raise ValueError('Error in reading disorder file: {}.'.format(self.args.muVar_fn))
            assert len(self.muVar_list)==self.wire_num, 'The length of muVar_list ({}) is not equal to the number of the unit cell of wire ({}).'.format(len(self.muVar_list),self.wire_num)
        else:  
            rng_muVar=np.random.default_rng(self.args.muVar_seed)          
            self.muVar_list=rng_muVar.normal(size=self.wire_num)*self.args.muVar

        if len(self.args.random_fn)>0:
            # print('Use random file:'.format(self.args.random_fn))
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
        'exp': lambda x: self.args.pot_peak*(np.exp(-(x-self.args.pot_peak_pos)**2/(2*self.args.pot_sigma**2))),
        'exp2': lambda x: self.args.pot_peak*(np.exp(-(x-self.args.pot_peak_pos)**2/(2*self.args.pot_sigma**2)))+self.args.pot_peak_R*(np.exp(-(x-self.args.pot_peak_pos_R)**2/(2*self.args.pot_sigma_R**2)))        
        }
        assert self.args.pot_type in self._potential, 'Potential type ({}) is not defined.'.format(self.args.pot_type)

    def get_potential_type(self):
        '''
        Plot inhomogeneous potential. Save the figure as png with potential name.
        '''
        y=self._potential[self.args.pot_type](np.linspace(0,self.args.L,self.wire_num))
        fig,ax=plt.subplots()
        ax.plot(np.arange(self.wire_num)*self.args.a/1000,y)
        ax.set_title(self.args.pot_type)
        ax.set_xlabel('L($\mu$m)')
        ax.set_ylabel('V(meV)')
        fig.savefig('{}.png'.format(self.args.pot_type))

    def get_hamiltonian_bare(self):
        '''
        Get the Hamiltonian of the nanowire nanowire, i.e., without barrier and lead.
        
        Returns
        -------
        kwant.builder.Builder
            The object that is to be `finalized`.
        '''
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
        '''
        Generate the spatial chemical potential considering inhomogeneous potential and disorder.
        '''
        self.mu_list=self.args.mu-self._potential[self.args.pot_type](np.linspace(0,self.args.L,self.wire_num))
        self.mu_list=self.mu_list-self.muVar_list

    def _SC_Delta_list(self):
        '''
        Generate the spatial random disorder in the parent superconductor.
        '''
        self.Delta0_list=self.args.Delta0*np.ones(self.wire_num) if self.args.DeltaVar==0 else self.random_list
        self.coupling_SC_SM_list=self.args.coupling_SC_SM*np.ones(self.wire_num) if self.args.coupling_SC_SM_Var==0 else self.random_list
        self.Delta_list=np.zeros(self.wire_num) if self.args.Vz>self.args.Vzc else self.Delta0_list*np.sqrt(1-(self.args.Vz/self.args.Vzc)**2)
        self.SC_Delta_list=[-coupling_SC_SM*(self.args.V_bias*t0s0+Delta*txs0)/np.sqrt(Delta**2-self.args.V_bias**2-(2*(self.args.V_bias>=0)-1)*1e-9j)*(Delta0>0) for Delta, Delta0, coupling_SC_SM  in zip(self.Delta_list,self.Delta0_list,self.coupling_SC_SM_list)] if self.args.SE else [Delta*txs0 for Delta in self.Delta_list]
    
    def _Vz_list(self):
        '''
        Generate the spatial random disorder in the effective g factor.'''
        self.Vz_list=self.args.Vz*np.ones(self.wire_num) if self.args.gVar==0 else self.random_list*self.args.Vz

    def _QD(self):
        '''
        Generate the quantum dot chemical potential, and remove the superconducting term in the quantum dot.
        '''
        if self.args.QD:
            for x in range(self.QD_num):
                self.hamiltonian_bare[self.lat(x)]=(2*self.t-self.args.mu+self.args.QD_peak*np.exp(-(x*self.args.a*1e-3)**2/self.args.QD_L**2))*tzs0+self.Vz_list[x]*t0sx-1j*self.args.dissipation*t0s0
            for x in range(self.QD_num_R):
                self.hamiltonian_bare[self.lat(self.wire_num-x-1)]=(2*self.t-self.args.mu+self.args.QD_peak_R*np.exp(-(x*self.args.a*1e-3)**2/self.args.QD_L_R**2))*tzs0+self.Vz_list[x]*t0sx-1j*self.args.dissipation*t0s0
        
    def _lead(self,junction,lead_pos,zero_barrier):
        '''
        Generate the Hamiltonian of the lead and attach it to the bare Hamiltonian of nanowire.
        
        Parameters
        ----------
        junction : kwant.builder.Builder
                The bare Hamiltonian to be attached.

        lead_pos : {'L','R'}
                The position of lead to be attached.

        zero_barrier : bool
                Whether zero barrier should be set. This is set to True for calculation of topological visibility.
        
        Returns
        -------
        junction : kwant.builder.Builder
                The Hamiltonian after attaching the lead.
        '''
        if self.args.barrier_relative is not None:
            self.args.barrier_E=self.args.mu+self.args.barrier_relative
        if zero_barrier:
            barrier_E=0
        else:
            barrier_E=self.args.barrier_E
        if lead_pos=='L':
            junction[(self.lat(x) for x in range(self.args.barrier_num))]=(2*self.t-self.args.mu+barrier_E)*tzs0+ self.args.Vz*t0sx
            sym_L=kwant.TranslationalSymmetry([-self.args.a])
            lead_L=kwant.Builder(sym_L,conservation_law=-tzs0)
            lead_L[self.lat(0)]=(2*self.t-self.args.mu_lead)*tzs0+self.args.Vz*t0sx
            lead_L[self.lat(0),self.lat(1)]=-self.t*tzs0-1j*self.alpha_R*tzsy
            junction.attach_lead(lead_L)
        elif lead_pos=='R':
            junction[(self.lat(self.wire_num-x-1) for x in range(self.args.barrier_num))]=(2*self.t-self.args.mu+barrier_E)*tzs0+ self.args.Vz*t0sx
            sym_R=kwant.TranslationalSymmetry([self.args.a])
            lead_R=kwant.Builder(sym_R,conservation_law=-tzs0)
            lead_R[self.lat(0)]=(2*self.t-self.args.mu_lead)*tzs0+self.args.Vz*t0sx
            lead_R[self.lat(0),self.lat(1)]=-self.t*tzs0-1j*self.alpha_R*tzsy
            junction.attach_lead(lead_R)
        return junction

    def get_hamiltonian_lead(self,lead_pos=None,zero_barrier=False):
        '''
        Combine the bare Hamiltonian for the nanowire and lead(s).
        
        Parameters
        ----------
        lead_pos : {'L','R','LR'}
                The position of lead to be attached. Both leads are attached for 'LR'.
        zero_barrier : bool, default=False
                Whether zero barrier should be set. This is set to True for calculation of topological visibility. The default is False.
        
        Returns
        -------
        kwant.builder.Builder
                The Hamiltonian combining the nanowire and lead, which are to be `finalized` next.
        '''
        if not hasattr(self, 'hamitonian_bare'):
            self.get_hamiltonian_bare()
        if self.args.lead_num==1:
            junction=self._lead(self.hamiltonian_bare,lead_pos,zero_barrier=zero_barrier)            
        elif self.args.lead_num==2:
            junction=self._lead(self.hamiltonian_bare,'L',zero_barrier=zero_barrier)
            junction=self._lead(self.hamiltonian_bare,'R',zero_barrier=zero_barrier)
        return junction

    def _Green_function(self,ham,delta=1e-3):
        '''
        Calculate the local density of state from the retarded Green's function, G= -1/pi* imag( w+i delta-H)^(-1).
        
        Parameters
        ----------
        ham : np.array
                The BdG Hamiltonian (4*wire_num,4*wire_num) in the form of np.array.
        delta : float, default=1e-3
                The inverse of lifetime, which is a positive infinitesimal in the Green's function to ensure the causality of the retarded Green's function.
        
        Returns
        -------
        np.array 
                The spatial local density of states (wire_num,1).
        '''
        GF=-1/np.pi*np.imag(np.linalg.inv((self.args.V_bias+1j*delta)*np.eye(ham.shape[0])-ham))
        return np.diag(GF).reshape((-1,4)).sum(axis=1)

    def LDOS(self,x,y):
        '''
        Get the local density of states using x and y, where y has to be `V_bias`. 
        
        Parameters
        ----------
        x : float
            Value of x-axis.
        y : float
            Value of y-axis (i.e., `V_bias` here).
        
        Returns
        -------
        np.array 
                The spatial local density of states (wire_num,1).
        '''
        setattr(self.args, self.args.x,x)
        setattr(self.args, self.args.y,y)

        # ugly implementation to enforce zero dissipation
        dissipation_tmp=self.args.dissipation
        self.args.dissipation=0
        ham=self.get_hamiltonian_bare().finalized()
        self.args.dissipation=dissipation_tmp
        return self._Green_function(ham.hamiltonian_submatrix())

    def _fix_phase(self,wf,pos):
        '''
        Fix the phase by setting the first element to real. The particle-hole symmetry opeartor is P=\sigma_y \otimes \tau_y K. Therefore, when applying to the Nambu basis, (u_up,u_down,v_down, v_up) |-> (-v*_up,v*_down,u*_down,-u*_up).
        
        Parameters
        ----------
        wf : np.array
                The BdG wave function (4*wire_num,1), where the basis is wire \otimes tau \otimes sigma.
        pos : bool
                positive energy if pos==True, else negative energy.
        
        Returns
        -------
        np.array 
            The BdG wave function (4*wire_num,1) after fixing the phase, where the basis is wire \otimes tau \otimes sigma.
        '''
        if pos:
            return wf*np.exp(-1j*np.angle(wf[0]))
        else:
            return -wf*np.exp(-1j*np.angle(wf[3]))

    def _sumindex(self,wf):
        '''
        Sum the four indices to get the spin average.
        
        Parameters
        ----------
        wf : np.array
            The BdG wave function (4*wire_num,1), where the basis is wire \otimes tau \otimes sigma.
        
        Returns
        -------
        np.array
            The spin averaged wave function (wire_num,1).
        '''
        return np.sum(np.abs(wf.reshape((-1,4)))**2,axis=1)

    def wavefunction(self,x,y):
        '''
        Get the wave functions at `x` that has an energy closest to `y` 
        
        Parameters
        ----------
        x : float
            Value of x-axis.
        y : float
            Value of y-axis (i.e., `V_bias` here).
        
        Returns
        -------
        dict 
            The dictionary for `val_p` (eigenvalues that is closest to `y`), `wf_p` (wave function in the BdG basis), `wf_1` and `wf_2 ` (wave functions in the Majorana basis), and `ansatz` (the initial starting point in y-axis).
        '''
        setattr(self.args, self.args.x,x)
        setattr(self.args, self.args.y,y)
        ham=self.get_hamiltonian_bare().finalized().hamiltonian_submatrix()
        vals_pos,vecs_pos=eigsh(ham,sigma=y)
        idx=np.abs(vals_pos-y).argmin()
        val_pos,vec_pos=vals_pos[idx],vecs_pos[:,idx]
        vec_neg=np.kron(np.eye(vec_pos.shape[0]//4),np.fliplr(np.diag([-1,1,1,-1])))@vec_pos.conj()

        vec_neg=self._fix_phase(vec_neg,False)
        vec_pos=self._fix_phase(vec_pos,True)

        vec_1=(vec_pos+vec_neg)/np.sqrt(2)
        vec_2=1j*(vec_pos-vec_neg)/np.sqrt(2)
        return {'val_p':val_pos,'wf_p':self._sumindex(vec_pos),'wf_1': self._sumindex(vec_1),'wf_2':self._sumindex(vec_2),'ansatz':y,'x':x}

    def ED(self,x,y):
        '''
        Get the eigenvalues from exact diagonlization.  Y should be `V_bias`, which is however ignored because the Hamiltonian does not depend on `V_bias`.
        
        Parameters
        ----------
        arg1 : type
                Description_of_arg1.
        
        Returns
        -------
        return1 : np.array
                All eigenvalues (4*wire_num,1).
        '''
        setattr(self.args, self.args.x,x)
        setattr(self.args, self.args.y,y)
        if abs(self.args.V_bias)<_eps:
            ham=self.get_hamiltonian_bare().finalized().hamiltonian_submatrix()
            return sorted(eigsh(ham,sigma=y,k=80,return_eigenvectors=False))
        else:
            return None

    def get_thermal(self,s_matrix,lead):
        '''
        Get the thermal conductance from `s_matrix` at the `lead`
        
        Parameters
        ----------
        s_matrix : np.array
                The s matrix of the NS juction.
        lead : {'L','R','LR','RL'}
                From which lead to measure. 
        
        Returns
        -------
        float
            The calculated conductance 
        '''
        incoming=0 if lead[1]=='L' else 1
        outgoing=0 if lead[0]=='L' else 1
        return s_matrix.transmission((outgoing,0),(incoming,0))+s_matrix.transmission((outgoing,1),(incoming,0))+s_matrix.transmission((outgoing,0),(incoming,1))+s_matrix.transmission((outgoing,1),(incoming,1))
    
    def get_conductance(self,s_matrix,lead):
        '''
        Get the conductance from `s_matrix` at the `lead`.
        
        Parameters
        ----------
        s_matrix : np.array
                The s matrix of the NS juction.
        lead : {'L','R','LR','RL'}
                From which lead to measure. 
        
        Returns
        -------
        float
            The calculated conductance.
        '''
        if len(lead)==1:
            return s_matrix.submatrix((0,0),(0,0)).shape[0]-s_matrix.transmission((0,0),(0,0))+s_matrix.transmission((0,1),(0,0))
        elif len(lead)==2:
            incoming=0 if lead[1]=='L' else 1
            outgoing=0 if lead[0]=='L' else 1
            if incoming==outgoing:
                # Reflection from the same lead
                return s_matrix.submatrix((incoming,0),(incoming,0)).shape[0]-s_matrix.transmission((incoming,0),(incoming,0))+s_matrix.transmission((incoming,1),(incoming,0))
            else:
                # Transmission through two leads
                return s_matrix.transmission((outgoing,0),(incoming,0))-s_matrix.transmission((outgoing,1),(incoming,0))
        else:
            raise ValueError('The number of lead ({}) should be 1 or 2'.format(len(lead)))

    def conductance(self,x,y):
        '''
        Calculate the local, nonlocal, thermal conductanc, topological visibility at a given `x` and `y`.
        
        Parameters
        ----------
        x : float
            Value of x-axis.
        y : float
            Value of y-axis.
        
        Returns
        -------
        G : dict
            The local (nonlocal) conductance with possible L, R, LR, and RL.
        TV : dict
            The topological visibility from two ends with L andR.
        kappa : dict
            The thermal conductance from two ends with LR and RL.
        '''
        setattr(self.args, self.args.x,x)
        setattr(self.args, self.args.y,y)
        if self.args.lead_num==1:
            G={}
            for lead_pos in self.args.lead_pos:
                hamiltonian_lead=self.get_hamiltonian_lead(lead_pos).finalized()
                s_matrix=kwant.smatrix(hamiltonian_lead,self.args.V_bias,check_hermiticity=False)
                G[lead_pos]=self.get_conductance(s_matrix, lead_pos)
            TV,kappa=repeat(None,2)
        elif self.args.lead_num==2:
            hamiltonian_lead=self.get_hamiltonian_lead().finalized()
            s_matrix=kwant.smatrix(hamiltonian_lead,self.args.V_bias,check_hermiticity=False)
            G={lead:self.get_conductance(s_matrix, lead) for lead in ['LL','RR','LR','RL']}
            assert((self.args.y_max-self.args.y_min)/self.args.y_num>_eps,'The number of points on y-axis is too large. Use a smaller y_num.')
            if abs(self.args.V_bias)<_eps:
                TV=self.get_TV()
                kappa={lead:self.get_thermal(s_matrix, lead) for lead in ['LR','RL']}
            else:
                TV,kappa=repeat(None,2)
        return G,TV,kappa

    def get_TV(self):
        '''
        Get the topological visibility, where the barrier height should be set to zero.

        Returns
        -------
        TV : dict
            The topological visibility from two ends with L and R.
        '''
        # barrier is set to zero when calculated TV
        hamiltonian_lead=self.get_hamiltonian_lead(zero_barrier=True).finalized()
        s_matrix=kwant.smatrix(hamiltonian_lead,0,check_hermiticity=False)

        S=s_matrix.data
        basis_wf = s_matrix.lead_info[0].wave_functions
        normalize=[0,0,3,3,0,0,3,3]
        phase = np.array([(-1)**m*basis_wf[m,n]/abs(basis_wf[m,n]) for n,m in enumerate(normalize)])
        fixphase=np.conj(np.prod(phase[:4]))*np.prod(phase[4:])
        TVL,TVR=np.linalg.det(S[:4,:4]),np.linalg.det(S[4:,4:])
        assert (np.abs(TVL.imag)<_TV_eps and np.abs(TVR.imag)<_TV_eps),'TVL and TVR are not real with imag=({:e},{:e})'.format(TVL.imag,TVR.imag)
        TVL=np.real(fixphase*TVL)
        TVR=np.real(fixphase*TVR)
        TV={'L':TVL,'R':TVR}
        return TV

        



        
        



