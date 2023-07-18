#!/bin/sh
#SBATCH --job-name=demo_scannet    # Job name
#SBATCH --output=./slurm_jobs/job_%j.log           # Standard output and error log
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=60gb                     # Job memory request
#SBATCH --constraint=rtx_3090|rtx_2080                     # GPU types
#SBATCH --gpus=1                     # Job GPUs request
##SBATCH --nodelist=seti
##SBATCH --exclude=lothlann             # Exclude nodes
#SBATCH --cpus-per-task=4
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
    mode=demo \
    start_deform=True \
    finetune=True \
    data.n_views=1 \
    data.dataset=ScanNet \
    data.split_type=all \
    weight=outputs/ScanNet/train/2022-09-19/16-03-50/model_best.pth \
    optimizer.method=RMSprop \
    optimizer.lr=0.01 \
    scheduler.latent_input.milestones=[1200] \
    scheduler.latent_input.gamma=0.1 \
    demo.epochs=2000 \
    demo.batch_id=5 \
    demo.batch_num=6 \
    log.print_step=100 \
    log.if_wandb=False \
