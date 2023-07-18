#
# Copyright (c) 8.2022. Yinyu Nie
# License: MIT
#

python main.py \
    mode=demo \
    start_deform=True \
    finetune=True \
    data.n_views=1 \
    data.dataset=3D-Front \
    data.split_type=bed \
    weight=outputs/3D-Front/train/2022-09-06/02-37-24/model_best.pth \
    optimizer.method=RMSprop \
    optimizer.lr=0.01 \
    scheduler.latent_input.milestones=[1200] \
    scheduler.latent_input.gamma=0.1 \
    demo.epochs=2000
    demo.batch_id=0 \
    demo.batch_num=1 \
    log.print_step=100 \
    log.if_wandb=False