#!/bin/sh
#SBATCH --job-name=preprocess_living_room           # Job name
#SBATCH --output=./slurm_jobs/job_%j.log           # Standard output and error log
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=30gb                     # Job memory request
#SBATCH --gpus=0                     # Job GPUs request
##SBATCH --nodelist=seti
##SBATCH --exclude=lothlann             # Exclude nodes
#SBATCH --cpus-per-task=24
##SBATCH --mail-user=yinyu.nie@tum.de
#SBATCH --qos=normal
#SBATCH --partition=submit

# Default output information
date;hostname;pwd
echo "Job Name = $SLURM_JOB_NAME"

# Your code
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/rhome/ynie/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/rhome/ynie/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/rhome/ynie/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/rhome/ynie/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate ss
python utils/threed_front/1_process_viewdata.py --room_type living --n_processes 24
