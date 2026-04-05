#!/bin/bash
#SBATCH -J "AOLM"
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem-per-cpu=4000
#SBATCH -o output_bam_v4.out
#SBATCH --time="4-00:00:00"
#SBATCH -w gnode079

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"
./pipeline.sh

echo "Time at exit: $(date)"




