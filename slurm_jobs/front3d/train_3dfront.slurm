#!/bin/sh
#SBATCH --job-name=train_3dfront_bedroom    # Job name
#SBATCH --output=./slurm_jobs/job_%j.log           # Standard output and error log
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=200gb                     # Job memory request
#SBATCH --constraint=rtx_2080                     # GPU types
#SBATCH --gpus=4                     # Job GPUs request
##SBATCH --nodelist=seti
##SBATCH --exclude=lothlann             # Exclude nodes
#SBATCH --cpus-per-task=16
##SBATCH --mail-user=yinyu.nie@tum.de
#SBATCH --qos=deadline
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
    start_deform=True \
    resume=False \
    finetune=True \
    weight=['outputs/3D-Front/train/2022-11-06/18-26-19/model_best.pth'] \
    distributed.num_gpus=4 \
    data.dataset=3D-Front \
    data.n_views=20 \
    data.aug=False \
    data.downsample_ratio=4 \
    device.num_workers=16 \
    train.batch_size=16 \
    train.epochs=500 \
    train.freeze=[] \
    scheduler.latent_input.milestones=[300] \
    scheduler.generator.milestones=[300] \
    log.if_wandb=True \
    exp_name=train_3dfront_bedroom
