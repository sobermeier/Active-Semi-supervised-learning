import time

import numpy as np
import torch
import pandas as pd

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase


@AL_ALGORITHMS.register('entropy')
class Entropy(ActiveBase):
    def __init__(self, gpu):
        super().__init__(gpu)
        print("Entropy started", flush=True)

    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]
        x, y, embs, logits, probs, preds = self.get_x_y_embs_logits_probs_preds(clf, data_loaders["sequential_ulb_loader"])
        diff = len(y) - len(idxs_unlabeled)
        x = x[diff:]
        y = y[diff:]
        embs = embs[diff:]
        logits = logits[diff:]
        probs = probs[diff:]
        preds = preds[diff:]
        probs += 1e-8
        entropy = - (torch.tensor(probs) * torch.log(torch.tensor(probs))).sum(1)
        entropy_sorted, idx_sorted = entropy.sort(descending=True)
        query_idx = idxs_unlabeled[idx_sorted[:n]]

        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[list(idx_sorted[:n])] = True

        correct = preds == y

        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), y, preds, correct, entropy.cpu().numpy(), query_idx],
            index=["id", "chosen", "label", "pred", "correct", "entropy", "query_idx"]).T
        return query_idx, query_df
