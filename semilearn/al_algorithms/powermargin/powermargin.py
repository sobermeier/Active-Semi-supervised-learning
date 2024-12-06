import time

import numpy as np
import pandas as pd
import torch

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase

# This implementation was based on https://github.com/ValentinMargraf/ActiveLearningPipelines/tree/main

@AL_ALGORITHMS.register('powermargin')
class PowerMargin(ActiveBase):
    def __init__(self, args, gpu):
        print("Power Margin initialized")
        super().__init__(args, gpu)
        self.seed = args.seed

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
        margin = (probs_sorted[:, 0] - probs_sorted[:,1]).cpu().numpy()
        margin_inv = 1-margin
        np.random.seed(self.seed)
        log_margin = np.log(margin_inv + 1e-8)
        rand = np.random.gumbel(loc=0, scale=1, size=len(idxs_unlabeled))
        pow_margin = log_margin + rand

        _, idx_sorted = torch.tensor(pow_margin).sort(descending=True)

        query_idx = idxs_unlabeled[idx_sorted[:n]]

        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[list(idx_sorted[:n])] = True

        correct = preds == y

        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), y, preds, correct, margin, margin_inv, log_margin, rand, pow_margin, query_idx],
            index=["id", "chosen", "label", "pred", "correct", "margin", "1 - margin", "log margin", "rand", "power margin", "query_idx"]).T

        return query_idx, query_df
