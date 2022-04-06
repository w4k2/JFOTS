import os
import datasets

if __name__ == '__main__':
    for fold in range(10):
        for dataset_name in datasets.names():
            command = f'sbatch ./run_slurm.sh -fold {fold} -dataset_name {dataset_name}'
            os.system(command)
