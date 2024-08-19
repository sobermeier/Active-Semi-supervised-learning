import time

import numpy as np
import pandas as pd
import torch

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase


@AL_ALGORITHMS.register('margin')
class Margin(ActiveBase):
    def __init__(self, args, gpu):
        print("Margin initialized")
        super().__init__(args, gpu)

    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]
        x, y, embs, logits, probs, preds = self.get_x_y_embs_logits_probs_preds(clf, data_loaders["sequential_ulb_loader"])
        diff = len(y) - len(idxs_unlabeled)
        y = y[diff:]
        probs = probs[diff:]
        preds = preds[diff:]

        print("probs", probs[0], flush=True)
        probs_sorted, idxs = torch.tensor(probs).sort(descending=True)
        print("probs_sorted", probs_sorted[0], idxs, flush=True)
        margin = probs_sorted[:, 0] - probs_sorted[:,1]
        _, idx_sorted = margin.sort(descending=False)

        query_idx = idxs_unlabeled[idx_sorted[:n]]

        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[list(idx_sorted[:n])] = True

        correct = preds == y

        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), y, preds, correct, margin.cpu().numpy(), query_idx],
            index=["id", "chosen", "label", "pred", "correct", "margin", "query_idx"]).T

        return query_idx, query_df
