# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer_single_gpu import Trainer
from options import MonodepthOptions
import os
import torch
import numpy as np
options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    seed = 3407
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    trainer = Trainer(opts)
    trainer.train()

