import os
import datasets

if __name__ == '__main__':
    datasets = ["haberman", "glass0"]
    for fold in range(2):
        for dataset_name in datasets:
            command = f'sbatch run_slurm.sh experiment.py -fold {fold} -dataset_name {dataset_name}'
            os.system(command)
