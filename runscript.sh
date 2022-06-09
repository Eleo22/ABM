#!/bin/bash
for i in {2..5}
do
        curr_folder="R0=$i"
        cd $curr_folder
        sbatch launchrgt0.5.sh
        sbatch launchrran.sh
        cd ..
done
