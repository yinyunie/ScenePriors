#!/bin/sh
#SBATCH --job-name=scannet_finetune_train    # Job name
#SBATCH --output=./slurm_jobs/job_%j.log           # Standard output and error log
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=300gb                     # Job memory request
#SBATCH --constraint=rtx_a6000|rtx_3090|rtx_2080                   # GPU types
#SBATCH --gpus=1                     # Job GPUs request
##SBATCH --nodelist=seti
##SBATCH --exclude=lothlann             # Exclude nodes
#SBATCH --cpus-per-task=4
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
python main.py \
    mode=test \
    start_deform=True \
    finetune=True \
    weight=outputs/ScanNet/train/2022-08-31/08-46-00/model_best.pth \
    distributed.num_gpus=1 \
    device.num_workers=4 \
    data.dataset=ScanNet \
    data.split_type=all \
    data.split_dir=splits \
    data.n_views=1 \
    optimizer.lr=0.0001 \
    scheduler.latent_input.milestones=[400] \
    scheduler.latent_input.gamma=0.1 \
    data.split_dir=splits \
    test.finetune_split=train \
    train.batch_size=4 \
    test.n_views_for_finetune=1 \
    test.epochs=400 \
    generation.dump_results=False \
    log.if_wandb=True \
    exp_name="scannet_finetune_train"