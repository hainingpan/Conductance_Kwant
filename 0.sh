#!/bin/bash
#SBATCH --mail-user=hnpan@terpmail.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --share
#SBATCH -t 20:05:00
#SBATCH -n 256

. ~/.profile

#cd /lustre/hnpan/multiband_kwant
cd $PWD

module load python/2.7.8
module load openmpi/gnu

mpirun python Majorana_main.py 
