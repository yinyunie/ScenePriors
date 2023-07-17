#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT

from . import ours
from . import loss

method_paths = {
    'Ours': ours
}

__all__ = ['method_paths']