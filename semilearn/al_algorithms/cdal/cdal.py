import time

import numpy as np
import pandas as pd
import torch

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase

import torch.nn.functional as F

from tqdm import tqdm

def kl_pairwise_distances(a, b):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)

    dist = np.zeros((a.size(0), b.size(0)), dtype=float)
    for i in range(b.size(0)):
        b_i = b[i]
        kl1 = a * torch.log(a / b_i)
        kl2 = b_i * torch.log(b_i / a)
        dist[:, i] = 0.5 * (torch.sum(kl1, dim=1)) + 0.5 * (torch.sum(kl2, dim=1))
    return dist

# This implementation originated from https://github.com/AminParvaneh/alpha_mix_active_learning/blob/main/query_strategies/cdal_sampling.py
@AL_ALGORITHMS.register('cdal')
class CDAL(ActiveBase):
    def __init__(self, args, gpu):
        print("CDAL initialized")
        super().__init__(args, gpu)


    def select_coreset(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = kl_pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        print(min_dist)

        idxs = []
        min_max_dists = []
        print('selecting coreset...')
        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            min_max_dists.append(min_dist[idx])
            dist_new_ctr = kl_pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        for i in range(len(idxs)):
            min_dist[idxs[i]] = min_max_dists[i]
        return idxs, min_dist

    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]

        x, y, embs, logits, probs, preds = self.get_x_y_embs_logits_probs_preds(clf, data_loaders["sequential_ulb_loader"])
        diff = len(y) - len(idxs_unlabeled)
        lb_mask = np.zeros(len(y), dtype=bool)
        lb_mask[0:diff] = True

        probs_ul = probs[diff:]
        probs_lb = probs[0:diff]

        # not sure why they do double softmax here
        chosen, min_dist = self.select_coreset(F.softmax(torch.tensor(probs_ul), dim=1).numpy(), F.softmax(torch.tensor(probs_lb), dim=1).numpy(), n)
        #chosen, min_dist = self.select_coreset(probs_ul, probs_lb, n)

        query_idx = idxs_unlabeled[chosen]


        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[chosen] = True

        y = y[diff:]
        preds = preds[diff:]

        correct = preds == y

        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), y, preds, correct, min_dist, query_idx],
            index=["id", "chosen", "label", "pred", "correct", "min_dist", "query_idx"]).T

        return query_idx, query_df
