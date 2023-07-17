#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

from .dataloader import our_dataloader
from .testing import Tester
from .training import Trainer

def get_trainer(cfg, net, optimizer, device=None):
    return Trainer(cfg=cfg, net=net, optimizer=optimizer, device=device)


def get_tester(cfg, net, device=None):
    return Tester(cfg=cfg, net=net, device=device)


def get_dataloader(cfg, mode):
    return our_dataloader(cfg=cfg, mode=mode)
