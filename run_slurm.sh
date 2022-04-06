#!/bin/bash

#SBATCH -P plgrid
#SBATCH -A plgjoannagrzyb2022a
#SBATCH -J eksperymentJFOTS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=joanna.grzyb@pwr.edu.pl

module add plgrid/tools/python/3.8.5
python3 -W ignore experiment.py
