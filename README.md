# 1D Semiconductor-superconductor nanowire
This repository summarizes all the codes in my Majorana-related works during my Ph.D.

# Usage
`python Majorana_main.py -mu=1 -L=1 -Delta0=0.2 -cond -LDOS` calculates the conductance (`-cond`) and LDOS (`-LDOS`) of a nanowire with wire length being 1 micron, chemical potential in the semiconductor being 1 meV, and superconductor gap being 0.2 meV. 

To run it in parallel using `$N` cores, use `mpirun -np $N_cores python -m mpi4py.futures Majorana_main.py -mu=1 -L=1 -Delta0=0.2 -cond -LDOS`, where `$N` should be replaced by the number of cores available.

## Output
### `-cond`
* `-lead_num=1` `-lead_pos=L` or `-lead_pos=R`: conductance.
* `-lead_num=1` `-lead_pos=LR`: conductance from the left end and right end, separately.
* `-lead_num=2` `-lead_pos=LR`: nonlocal conductance, topological visibility, thermal conductance (`kappa`)
* Example: `mpirun -np 4 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=2 -lead_pos=LR -cond`

### `-LDOS`
* Local density of states as a function of `x` and `y` where `y` has to be `V_bias`. The third axis is the spatial position.

* Example: `mpirun -np 4 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=2 -lead_pos=LR -LDOS -barrier_E=0 -dissipation=0`

### `-energy`
* Calculate the energy spectrum. If the self-energy `SE` is not flagged, exact diagonalization is used. Otherwise, the energy spectrum is obtained by first calculating the LDOS, and exacting the peaks.

* Example (No self-energy): `mpirun -np 4 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=2 -lead_pos=LR -energy -barrier_E=0 -dissipation=0`

* Example (Self-energy): ` mpirun -np 4 python -m mpi4py.futures Majorana_main.py -L=1 -mu=1 -lead_num=2 -lead_pos=LR -energy -barrier_E=0 -dissipation=0 -SE`

### `-wavefunction`
* Wavefunction is better visualized using `jupyter notebook`. An example if provided in `wavefunction.ipynb`.

# Help
`python Majorana_main.py -h` shows the definitions for all parameters.

## Imhomogeneous Potential type
*NB: Some rare types of inhomogeneous potentials have been removed in the newer branch `refactoring-2`.*
![`cos`](cos.png)
![`exp`](exp.png)
![`exp2`](exp2.png)

# Requirements
[`kwant`](https://kwant-project.org/), 
[`mpi4py`](https://mpi4py.readthedocs.io/en/stable/install.html), 
[`tinyarray`](https://pypi.org/project/tinyarray/), 
[`SciPy`](https://scipy.org/),
[`NumPy`](https://numpy.org/),
[`matplotlib`](https://matplotlib.org/),


# To Do:
- [ ] Extend y-axis beyond the energy (bias voltage) in calculating the LDOS
- [x] Incorporate the finite temperature (only support when y is v_bias)
- [x] Add the wavefunction
- [x] Energy spectrum
