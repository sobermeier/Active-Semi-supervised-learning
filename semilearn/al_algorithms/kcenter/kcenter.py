import time

import numpy as np
import pandas as pd
import torch

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase

from tqdm import tqdm

# This implementation originated from https://github.com/ej0cl6/deep-active-learning/blob/master/query_strategies/kcenter_greedy.py
# MIT License
#
# Copyright (c) 2021 Natural Language Processing @UCLA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
@AL_ALGORITHMS.register('kcenter')
class KCenter(ActiveBase):
    def __init__(self, args, gpu):
        print("kCenter initialized")
        super().__init__(args, gpu)

    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]

        x, y, embs, logits, probs, preds = self.get_x_y_embs_logits_probs_preds(clf, data_loaders["sequential_ulb_loader"])
        diff = len(y) - len(idxs_unlabeled)
        lb_mask = np.zeros(len(y), dtype=bool)
        lb_mask[0:diff] = True



        dist_mat = np.matmul(embs, embs.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(lb_mask), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~lb_mask, :][:, lb_mask]
        chosen = []

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(len(y))[~lb_mask][q_idx_]
            lb_mask[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~lb_mask, q_idx][:, None], axis=1)

        chosen = np.where(lb_mask[diff:])
        query_idx = idxs_unlabeled[chosen]


        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[chosen] = True


        lb_mask_old = np.zeros(len(y), dtype=bool)
        lb_mask_old[0:diff] = True

        x = x[diff:]
        y = y[diff:]
        embs = embs[diff:]
        logits = logits[diff:]
        probs = probs[diff:]
        preds = preds[diff:]

        correct = preds == y


        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), y, preds, correct, lb_mask[diff:], np.min(dist_mat[~lb_mask_old, :], axis=1), query_idx],
            index=["id", "chosen", "label", "pred", "correct", "lb_mask", "dist_mat", "query_idx"]).T

        return query_idx, query_df
