import time

import numpy as np

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase


@AL_ALGORITHMS.register('random')
class Random(ActiveBase):
    def __init__(self, gpu):
        super().__init__(gpu)

    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]
        chosen = np.random.choice(idxs_unlabeled, n, replace=False)
        return chosen
