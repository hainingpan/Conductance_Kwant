# Semiconductor-superconductor nanowire
This repository summarizes all the codes in my Majorana-related works during my Ph.D.

# Usage
`python Majorana_main.py -mu=1 -L=1 -Delta0=0.2 -cond -LDOS` calculates the conductance (`-cond`) and LDOS (`-LDOS`) of a nanowire with wire length being 1 micron, chemical potential in the semiconductor being 1 meV, and superconductor gap being 0.2 meV. 

To run it in parallel, use `mpirun -np 32 python -m mpi4py.futures Majorana_main.py -mu=1 -L=1 -Delta0=0.2 -cond -LDOS`, where 32 cores are used.

## Output
### `-cond`
* `-lead_num=1` `-lead_pos=L` or `-lead_pos=R`: conductance.
* `-lead_num=1` `-lead_pos=LR`: conductance from the left end and right end, separately.
* `-lead_num=2` `-lead_pos=LR`: nonlocal conductance, topological visibility (`TVL`, `TVR`), thermal conductance (`kappa`)

### `-LDOS`
* Local density of states as a function of `x` and `y` where `y` has to be `V_bias`. The third axis is the spatial position.

### `-wavefunction`
* Wavefunction is better visualized using `jupyter notebook`.

# Help
`python Majorana_main.py -h` shows the definitions for all parameters.

## Imhomogeneous Potential type
*NB: Some rare types of inhomogeneous potentials have been removed in the newer branch `refactoring-2`.*
![`cos`](cos.png)
![`exp`](exp.png)
![`exp2`](exp2.png)

# Requirements
[`kwant`](https://kwant-project.org/), [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/install.html), [`tinyarray`](https://pypi.org/project/tinyarray/)


# To Do:
- [ ] Extend y-axis beyond the energy (bias voltage) in calculating the LDOS
- [ ] Incorporate the finite temperature (only support when y is v_bias)
- [ ] Add the wavefunction

# Legacy

This code is expanded for completeness. The previous version can be found in the directory `legacy` where the previous `README.md` is as follows for record-keeping purposes. The major difference in the naming rule is also tracked as

|old|new|
|---|---|
|alpha_R|alpha|
|delta0|Delta0|
|wireLength|L|
|muLead|mu_lead|
|barrierE|barrier_E|
|isDissipationVar | |
|isQD|QD|
|qdPeak|QD_peak|
|qdLength|QD_L|
|qdPeakR|QD_peak_R|
|qdLengthR|QD_L_R|
|couplingSCSM|coupling_SC_SM|
|Vc|Vzc|
|vz|Vz|
|v_bias|V_bias|
|potType|pot_type|
|potPeakPos|pot_peak_pos|
|potSigma|pot_sigma|
|potPeak|pot_peak|
|potPeakPosR|pot_peak_pos_R|
|potSigmaR|pot_sigma_R|
|potPeakR|pot_peak_R|
|muVarList|muVar_fn|
|N_muVar||
|scatterList||
|muVarType||
|randList|random_fn|
|deltaVar|DeltaVar|
|couplingSCSMVar|coupling_SC_SM_Var|
|leadPos|lead_pos|
|leadNum|lead_num|
|isS|(True)|
|xNum|x_num|
|yNum|y_num|
|xMin|x_min|
|xMax|x_max|
|yMin|y_min|
|yMax|y_max|
|xNum|x_num|
|yNum|y_num|
|xUnit|x_unit|
|yUnit|y_unit|
|colortheme|cmap|
|error||

>*Example:*
>
>
>
>* `python Majorana_main.py wireLength=100`  Run on single-core, for a nanowire with 100 sites where lattice constant is 10nm. So the total length is 1um. * `mpiexec -n 4 python -m mpi4py.futures Majorana_adaptive.py wireLength=100` Run on local MPI executor pool. using adaptive sampling instead of uniform grid sampling* `a=1`    lattice constance, unit is 10nm
>
>* `alpha_R=5`    Spin-orbital coupling, unit is eV*nm
>* `mpirun -n 4 python Majorana_main.py wireLength=100 leadNum=2` Run on local multiple cores, for a 1 um nanowire with both lead attached to calculate nonlocal conductance
>* `wireLength=1000`    number of sites in the 1d nanowire
>* 0.sh for submitting to Deepthought2 cluster*Current utitlity with the default value:*
>* `barrierNum=2`    number of barrier
>* `isTV=0(1)`    whether export Topological visibility 
>* `dissipation=0.0001`   dissipation, unit is meV
>
>* `isQD=0(1)`  whether applying quantum dots(QD)
>* `mu=0.2`    chemical potential in the semiconductor nanowire, unit is meV
>* `qdLength=20`    length of QD, unit is number of sites
>* `delta0=0.2`   superconducting gap of covering superconductor
>* `couplingSCSM=0.2`    coupling strength of covering superconductor(SC) and semiconductor(SM), unit is meV
>* `muLead=25`    chemical potential in the lead
>* `potType=0/sin/sintheta/cos/sin2/sinabs/lorentz/lorentzsigmoid/exp/sigmoid`  The shape of inhomogeneous potential, definition see Majorana_module.py. 0 is for constant chemical potential.
>* `barrierE=10`    energy of barrier, unit is meV
>* `potSigma=0`    sigma of Gaussian potential, unit is number of sites
>* `isDissipation=0(1)`    whether using Vz-dependent dissipation    
>* `muVar=0`    variance of disorder in chemical potential, unit is meV
>* `qdPeak=0.4`    peak of QD, unit is meV
>* `gVar=0`    variance of disorder in effective g factor, unit is meV
>* `isSE=0(1)`    whether applying self-energy
>* `deltaVar=0` variance of disorder in SC, unit is meV
>* `vc=0`    the Zeeman field that covering SC gap collapse, unit is meV
>* `vz0=0`    initial point of Zeeman field range
>* `potPeakPos=0`    position of potential peak
>* `vBiasNum=1001`    number of bias voltage
>* `potPeak=0`    peak of potential, unit is meV
>* `vzMax=1.024`    maximum of Zeeman field range(only in adaptive currently)
>* `muVarList=0`    random profile of disorder in chemical potential, accept file
>* `leadPos=0/1`    attach lead to the left or right
>* `randList=0`    random profile of disorder, accept file
>* `muStep=0.002`    step of chemical potential range
>* `vz=0`    Zeeman field
>* `isMu=0`    determine x-axis is Zeeman field or chemical potential(only in adaptive currently)
>* `vBias=256`    bias voltage
>
>* `vzStep=0.002`    step of Zeeman field range
>* `error=0`    error message
>* `muMax=1`    maximum of chemical potential range
>* `leadNum=1`    number of leads
>* `muNum=0`    number of chemical potential


<!-- 
things to revert:
1. Change t from 25
2. Change the xNum in the old one to the one without -1 -->





