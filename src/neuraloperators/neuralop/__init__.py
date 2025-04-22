'''
The neuraloperators folder is based on https://github.com/neuraloperator/neuraloperator
'''
__version__ = '0.2.1'

from . import datasets, mpu
from .losses import (BurgersEqnLoss, H1Loss, ICLoss, LpLoss, MSEloss,
                     WeightedSumLoss)
from .models import TFNO, TFNO2d, get_model
from .training import CheckpointCallback, Trainer
