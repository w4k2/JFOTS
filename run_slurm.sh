#!/bin/bash

#SBATCH -P plgrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH -A plgjoannagrzyb2021a
#SBATCH --mail-type=END
#SBATCH --mail-user=joanna.grzyb@pwr.edu.pl


python3 -W ignore experiment.py ${@:2}
