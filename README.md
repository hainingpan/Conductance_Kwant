# Conductance_Kwant

*Example:*

* `python Majorana_main.py wireLength=100`  Run on single core, for a nanowire with 100 sites where lattice constant is 10nm. So the total length is 1um. 
* `mpirun -n 4 python Majorana_main.py wireLength=100 leadNum=2` Run on local multiple cores, for a 1 um nanowire with both lead attached to calculate nonlocal conductance
* `mpiexec -n 4 python -m mpi4py.futures Majorana_adaptive.py wireLength=100` Run on local MPI executor pool. using adaptive sampling instead of uniform grid sampling
* 0.sh for submitting to Deepthought2 cluster

*Current utitlity with the default value:*

* `isTV=0(1)`    whether export Topological visibility 
* `a=1`    lattice constance, unit is 10nm
* `mu=0.2`    chemical potential in the semiconductor nanowire, unit is meV
* `alpha_R=5`    Spin-orbital coupling, unit is eV*nm
* `delta0=0.2`   superconducting gap of covering superconductor
* `wireLength=1000`    number of sites in the 1d nanowire
* `muLead=25`    chemical potential in the lead
* `barrierNum=2`    number of barrier
* `barrierE=10`    energy of barrier, unit is meV
* `dissipation=0.0001`   dissipation, unit is meV
* `isDissipation=0(1)`    whether using Vz-dependent dissipation    
* `isQD=0(1)`  whether applying quantum dots(QD)
* `qdPeak=0.4`    peak of QD, unit is meV
* `qdLength=20`    length of QD, unit is number of sites
* `isSE=0(1)`    whether applying self-energy
* `couplingSCSM=0.2`    coupling strength of covering superconductor(SC) and semiconductor(SM), unit is meV
* `vc=0`    the Zeeman field that covering SC gap collapse, unit is meV
* `potType=0/sin/sintheta/cos/sin2/sinabs/lorentz/lorentzsigmoid/exp/sigmoid`  The shape of inhomogeneous potential, definition see Majorana_module.py. 0 is for constant chemical potential.
* `potPeakPos=0`    position of potential peak
* `potSigma=0`    sigma of Gaussian potential, unit is number of sites
* `potPeak=0`    peak of potential, unit is meV
* `muVar=0`    variance of disorder in chemical potential, unit is meV
* `muVarList=0`    random profile of disorder in chemical potential, accept file
* `gVar=0`    variance of disorder in effective g factor, unit is meV
* `randList=0`    random profile of disorder, accept file
* `deltaVar=0` variance of disorder in SC, unit is meV
* `vz=0`    Zeeman field
* `vz0=0`    initial point of Zeeman field range
* `vBias=256`    bias voltage
* `vBiasNum=1001`    number of bias voltage
* `vzStep=0.002`    step of Zeeman field range
* `vzMax=1.024`    maximum of Zeeman field range(only in adaptive currently)
* `muMax=1`    maximum of chemical potential range
* `leadPos=0/1`    attach lead to the left or right
* `leadNum=1`    number of leads
* `muStep=0.002`    step of chemical potential range
* `muNum=0`    number of chemical potential
* `isMu=0`    determine x-axis is Zeeman field or chemical potential(only in adaptive currently)
* `error=0`    error message
