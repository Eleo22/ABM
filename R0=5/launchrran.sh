#!/bin/sh

# Redirect output with: -o <file_name>
# Redirect error messages with: -e <file_name>
# Specify memory via 1: --mem=<memory in megabytes>
#                    2: --mem-per-cpu=<memory in megabytes>
# Indicate number of CPUs via: --cpus-per-task=<number_of_cpus>
# To receive email when job is done use:
#   --mail-user=<your_email> --mail-type=END

#SBATCH -o print_rran.txt
#SBATCH -e code_err_rran.txt
#SBATCH -p mmmi -q mmmi
#SBATCH --mem=50000

srun python3 ABMrran.py

exit 0

