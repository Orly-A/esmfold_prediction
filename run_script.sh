#!/bin/sh
#SBATCH --time=96:00:00
#SBATCH --mem=30gb
#SBATCH -c 38

source /vol/ek/Home/orlyl02/working_dir/python3_venv/bin/activate.csh
python3 /vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/combine_esm_pkls.py > /vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/combine_log.log
