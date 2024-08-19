import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase

from tqdm import tqdm

# This implementation originated from https://github.com/JordanAsh/badge/blob/master/query_strategies/core_set.py
@AL_ALGORITHMS.register('coreset')
class Coreset(ActiveBase):
    def __init__(self, args, gpu):
        print("Coreset initialized")
        super().__init__(args, gpu)

    def furthest_first(self, X, X_set, n):
        print(np.shape(X), np.shape(X_set), flush=True)
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []
        min_max_dists = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            min_max_dists.append(min_dist[idx])
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
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

        chosen, min_dist = self.furthest_first(embs[~lb_mask, :], embs[lb_mask, :], n)

        query_idx = idxs_unlabeled[chosen]


        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[chosen] = True

        x = x[diff:]
        y = y[diff:]
        embs = embs[diff:]
        logits = logits[diff:]
        probs = probs[diff:]
        preds = preds[diff:]

        correct = preds == y

        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), y, preds, correct, lb_mask[diff:], min_dist, query_idx],
            index=["id", "chosen", "label", "pred", "correct", "lb_mask", "min_dist", "query_idx"]).T

        return query_idx, query_df
