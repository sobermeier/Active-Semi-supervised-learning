import time

import numpy as np
import torch
import pandas as pd
from scipy import stats

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase

# This implementation originated from https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
# Please cite the original paper if you use this method.

def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    dist = X1_norm_square * X2_norm_square + Y1_norm_square * Y2_norm_square - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
    # Numerical errors may cause the distance squared to be negative.
    assert np.min(dist) / np.max(dist) > -1e-4
    dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
    return dist


def init_centers(X1, X2, chosen, chosen_list,  mu, D2):
    if len(chosen) == 0:
        ind = np.argmax(X1[1] * X2[1])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
        D2[ind] = 0
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in chosen: ind = customDist.rvs(size=1)[0]
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    chosen.add(ind)
    chosen_list.append(ind)
    print("mu: ", str(len(mu)) + '\t DS sum: ' + str(sum(D2)), flush=True)
    return chosen, chosen_list, mu, D2

@AL_ALGORITHMS.register('badge')
class Badge(ActiveBase):
    def __init__(self, args, gpu):
        super().__init__(args, gpu)
        print("Badge started", flush=True)


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

        m = (~self.idxs_lb).sum()
        mu = None
        D2 = None
        chosen = set()
        chosen_list = []
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(probs, axis=-1)

        probs = -1 * probs
        probs[np.arange(m), max_inds] += 1
        prob_norms_square = np.sum(probs ** 2, axis=-1)

        for _ in range(n):
            chosen, chosen_list, mu, D2 = init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen,
                                                       chosen_list, mu, D2)

        query_idx = idxs_unlabeled[chosen_list]

        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[chosen_list] = True

        correct = preds == y

        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), y, preds, correct, mu, D2, query_idx],
            index=["id", "chosen", "label", "pred", "correct", "mu", "D2", "query_idx"]).T
        return query_idx, query_df
