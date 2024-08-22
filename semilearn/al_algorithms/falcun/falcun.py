import time

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import pairwise_distances

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase

from math import cos, pi

# This implementation originated from https://gitlab.lrz.de/ru96mop/ideal/-/blob/main/src/deepal/query_strategies/falcun.py

def _calc(r, R, max_gamma=1, min_gamma=0.1):
    """
    r=current round
    R=number of rounds
    lrs=starting gamma
    min_lrs=1  --> 0=random selection
    """
    return min_gamma + (max_gamma - min_gamma) * ((1 + cos(pi * r / R)) / 2)

@AL_ALGORITHMS.register('falcun')
class Falcun(ActiveBase):
    def __init__(self, args, gpu):
        super().__init__(args, gpu)
        print("FALCUN started", flush=True)
        self.seed = args.seed

        self.distance_metric = args.falcun_distance_metric
        self.gamma, self.gamma_start, self.gamma_min = args.falcun_gamma, args.falcun_gamma, args.falcun_gamma_min
        self.uncertainty = args.falcun_uncertainty
        print("unc", self.uncertainty)

    def get_unc(self, probs, uncertainty="margin"):
        probs_sorted, idxs = probs.sort(descending=True)
        if uncertainty == "margin":
            return (1 - (probs_sorted[:, 0] - probs_sorted[:, 1])).numpy(), idxs[:, 0]
        elif uncertainty == "entropy":
            probs += 1e-8
            entropy = -(probs * torch.log(probs)).sum(1)
            return ((entropy - entropy.min()) / (entropy.max() - entropy.min())).numpy(), idxs[:, 0]
        else:  # lc
            return (1 - probs_sorted[:, 0]).numpy(), idxs[:, 0]

    def get_indices(self, n, probs, dists, pred_labels):
        unc = dists.copy()
        weight_dist = 1
        ind_selected, vec_selected, class_selection, onehot_masks, probs_per_class = [], [], [], [], []
        unlabeled_range = np.arange(len(dists))
        candidate_mask = np.ones(len(dists), dtype=bool)
        for label in range(self.args.num_classes):
            class_mask = np.zeros(len(dists), dtype=bool)
            class_mask[pred_labels == label] = True
            onehot_masks.append(class_mask)
            probs_per_class.append(probs[class_mask])

        while len(vec_selected) < n:
            if len(vec_selected) > 0:
                current_probs = probs_per_class[class_selection[-1]]
                current_mask = onehot_masks[class_selection[-1]]
                new_dists = pairwise_distances(current_probs, [vec_selected[-1]],
                                               metric=self.distance_metric).ravel().astype(float)
                dists[current_mask] = np.minimum.reduce([dists[current_mask], new_dists])
            x = dists[candidate_mask]
            if sum(x) > 0:
                # unc_probs = unc[candidate_mask] / sum(unc[candidate_mask])
                # dist_probs = dists[candidate_mask] / sum(dists[candidate_mask])
                # combined_probs = (weight_dist * np.array(dist_probs)) + ((1-weight_dist) * np.array(unc_probs))
                # combined_probs /= np.sum(combined_probs)
                # ind = np.random.choice(unlabeled_range[candidate_mask], size=1, p=combined_probs)[0]

                # unc_probs = (unc[candidate_mask]) ** (self.gamma_start - self.gamma) / sum(
                #     (unc[candidate_mask]) ** (self.gamma_start - self.gamma))
                #
                # dist_probs = (dists[candidate_mask]) ** self.gamma / sum((dists[candidate_mask]) ** self.gamma)
                # final_probs = (unc_probs * dist_probs) / sum(unc_probs * dist_probs)
                #
                # ind = np.random.choice(unlabeled_range[candidate_mask], size=1, p=final_probs)[0]

                x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))

                dist_probs = (x_norm + unc[candidate_mask]) ** self.gamma / sum(
                    (x_norm + unc[candidate_mask]) ** self.gamma)
                ind = np.random.choice(unlabeled_range[candidate_mask], size=1, p=dist_probs)[0]

            else:
                ind = np.random.choice(unlabeled_range[candidate_mask], size=1)[0]
            candidate_mask[ind] = False
            vec_selected.append(probs[ind])
            ind_selected.append(ind)
            class_selection.append(pred_labels[ind])
            # weight_dist = _calc(len(vec_selected), n, 1, 0.5)
            # self.gamma = _calc(len(vec_selected), n, self.gamma_start, self.gamma_min)
        return ind_selected



    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]
        x, y, embs, logits, probs, preds = self.get_x_y_embs_logits_probs_preds(clf, data_loaders["sequential_ulb_loader"])
        diff = len(y) - len(idxs_unlabeled)
        y = y[diff:]
        probs = probs[diff:]
        preds = preds[diff:]

        unc, preds2 = self.get_unc(torch.tensor(probs), uncertainty=self.uncertainty)
        unc_copy = unc.copy()
        selected = self.get_indices(n, probs, dists=unc, pred_labels=preds)

        query_idx = idxs_unlabeled[selected]

        probs_sorted, idxs = torch.tensor(probs).sort(descending=True)
        margin = 1 - probs_sorted[:, 0] - probs_sorted[:,1]

        confidence = (1 -probs_sorted[:, 0]).cpu().numpy()

        probs += 1e-8
        entropy = - (torch.tensor(probs) * torch.log(torch.tensor(probs))).sum(1)

        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[list(selected)] = True

        correct = preds == y

        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), y, preds, correct, unc, unc_copy, entropy.cpu().numpy(), margin.cpu().numpy(), confidence, query_idx],
            index=["id", "chosen", "label", "pred", "correct", "dist", "uncertainty",  "entropy", "inv margin", "inv confidence", "query_idx"]).T

        return query_idx, query_df
