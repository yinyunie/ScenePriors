#  Copyright (c) 1.2022. Yinyu Nie
#  License: MIT
import hydra
import os
import logging

def single_proc_run(config):
    from configs.config_utils import CONFIG
    from net_utils.distributed import initiate_environment
    initiate_environment(config)
    cfg = CONFIG(config)

    if config.mode == 'train':
        from train import Train
        trainer = Train(cfg=cfg)
        trainer.run()
    elif config.mode == 'test':
        from test import Test
        tester = Test(cfg=cfg)
        tester.run()
    elif config.mode == 'interpolation':
        from interpolation import Interpolation
        interpolater = Interpolation(cfg=cfg)
        interpolater.run()
    elif config.mode == 'generation':
        from generation import Generation
        generator = Generation(cfg=cfg)
        generator.run()
    elif config.mode == 'demo':
        from demo import Demo
        example = Demo(cfg=cfg)
        example.run()

@hydra.main(config_path='configs/config_files', config_name='default.yaml')
def main(config):
    os.environ["OMP_NUM_THREADS"] = str(config.distributed.OMP_NUM_THREADS)
    config.root_dir = hydra.utils.get_original_cwd()

    from net_utils.distributed import multi_proc_run
    logging.info('Initialize device environments')

    if config.distributed.num_gpus > 1:
        multi_proc_run(config.distributed.num_gpus, fun=single_proc_run, fun_args=(config,))
    else:
        single_proc_run(config)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()