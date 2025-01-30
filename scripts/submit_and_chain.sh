#!/bin/bash

# Define the base script and the new script
BASE_SCRIPT="/nfs-share/pa511/new_work/scripts/slurm_fl.mauao"
NEW_SCRIPT="/nfs-share/pa511/new_work/scripts/slurm_fl_temp.mauao"

# Copy the base script to the new script
cp $BASE_SCRIPT $NEW_SCRIPT

# Optionally modify the new script here if needed
# For example, you can uncomment additional parameters
# sed -i 's/# PARAMS="\$PARAMS simulation.warmup_ratio=0.05 simulation.cooloff_ratio=0.1"/PARAMS="\$PARAMS simulation.warmup_ratio=0.05 simulation.cooloff_ratio=0.1"/' $NEW_SCRIPT

# Submit the new script and capture the job ID
JOB_ID=$(sbatch $NEW_SCRIPT | awk '{print $4}')

# Use the captured job ID as a dependency for the next run
# For example, you can create another script and submit it with the dependency
NEXT_SCRIPT="/nfs-share/pa511/new_work/scripts/next_job.mauao"
cp $BASE_SCRIPT $NEXT_SCRIPT
sed -i "s/^#SBATCH -J .*/#SBATCH -J Next_Job/" $NEXT_SCRIPT
sed -i "s/^#SBATCH -w .*/#SBATCH -d afterok:$JOB_ID/" $NEXT_SCRIPT

# Submit the next script
sbatch $NEXT_SCRIPT
