#!/bin/sh
#SBATCH --job-name=pretrain_3dfront_bedroom    # Job name
#SBATCH --output=./slurm_jobs/job_%j.log           # Standard output and error log
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=300gb                     # Job memory request
#SBATCH --constraint=rtx_a6000|rtx_3090|rtx_2080                     # GPU types
#SBATCH --gpus=4                     # Job GPUs request
##SBATCH --nodelist=seti
##SBATCH --exclude=lothlann             # Exclude nodes
#SBATCH --cpus-per-task=32
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

conda activate sceneprior
python main.py \
    start_deform=False \
    resume=False \
    finetune=False \
    weight=[] \
    distributed.num_gpus=4 \
    data.dataset=3D-Front \
    data.split_type=bed \
    data.n_views=20 \
    data.aug=False \
    device.num_workers=32 \
    train.batch_size=128 \
    train.epochs=800 \
    train.freeze=[] \
    scheduler.latent_input.milestones=[400] \
    scheduler.generator.milestones=[400] \
    log.if_wandb=True \
    exp_name=pretrain_3dfront_bedroom
