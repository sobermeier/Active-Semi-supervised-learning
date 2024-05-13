import time

import numpy as np

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase


@AL_ALGORITHMS.register('entropy')
class Entropy(ActiveBase):
    def __init__(self, gpu):
        super().__init__(gpu)

    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]
        x, y, embs, logits, probs, preds = self.get_x_y_embs_logits_probs_preds(clf, data_loaders["train_ulb"])
        probs += 1e-8
        entropy = - (probs * torch.log(probs)).sum(1)
        _, idx = entropy.sort(descending=True)
        return idxs_unlabeled[idx[:n]]
