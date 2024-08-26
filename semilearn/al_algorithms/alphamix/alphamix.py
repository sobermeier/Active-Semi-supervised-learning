import copy
import math
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase

# This implementation originated from https://gitlab.lrz.de/ru96mop/ideal/-/blob/main/src/deepal/query_strategies/alpha_mix.py
@AL_ALGORITHMS.register('alphamix')
class AlphaMix(ActiveBase):
    def __init__(self, args, gpu):
        super().__init__(args, gpu)
        print("Alpha Mix started", flush=True)
        self.alpha_closed_form_approx =False
        self.alpha_opt =False
        self.alpha_cap =0.03125
        self.alpha_learning_rate =0.1
        self.alpha_clf_coef =1.0
        self.alpha_l2_coef =0.01
        self.alpha_learning_iters =5
        self.alpha_learn_batch_size =1000000

    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]

        x, y, embs, logits, probs, preds = self.get_x_y_embs_logits_probs_preds(clf, data_loaders["sequential_ulb_loader"])
        diff = len(y) - len(idxs_unlabeled)



        ulb_probs = probs[diff:]
        org_ulb_embedding = embs[diff:]

        probs_sorted, probs_sort_idxs = torch.tensor(ulb_probs).sort(descending=True)
        pred_1 = probs_sort_idxs[:, 0]

        # lb_probs = self.predict_prob(self.X[self.idxs_lb], self.Y[self.idxs_lb])
        org_lb_embedding = embs[:diff]
        y_lb = y[:diff]

        ulb_embedding = torch.tensor(org_ulb_embedding)
        lb_embedding = torch.tensor(org_lb_embedding)

        unlabeled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)


        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
        candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

        if self.alpha_closed_form_approx:
            var_emb = Variable(ulb_embedding, requires_grad=True).cuda(self.gpu)
            out = clf(var_emb, only_fc=True)
            loss = F.cross_entropy(out, pred_1.cuda(self.gpu))
            grads = torch.autograd.grad(loss, var_emb)[0].data.cpu()
            del loss, var_emb, out
        else:
            grads = None

        alpha_cap = 0.
        while alpha_cap < 1.0:
            alpha_cap += self.alpha_cap

            tmp_pred_change, tmp_min_alphas = \
                self.find_candidate_set(
                    lb_embedding, ulb_embedding, pred_1, torch.tensor(ulb_probs), alpha_cap=alpha_cap,
                    Y=torch.tensor(y_lb), clf=clf,
                    grads=grads)

            is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

            min_alphas[is_changed] = tmp_min_alphas[is_changed]
            candidate += tmp_pred_change

            print('With alpha_cap set to %f, number of inconsistencies: %d' % (
            alpha_cap, int(tmp_pred_change.sum().item())))

            if candidate.sum() > n:
                break

        if candidate.sum() > 0:
            print('Number of inconsistencies: %d' % (int(candidate.sum().item())))

            print('alpha_mean_mean: %f' % min_alphas[candidate].mean(dim=1).mean().item())
            print('alpha_std_mean: %f' % min_alphas[candidate].mean(dim=1).std().item())
            print('alpha_mean_std %f' % min_alphas[candidate].std(dim=1).mean().item())

            c_alpha = F.normalize(torch.tensor(org_ulb_embedding)[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()

            candidate_idxs = self.sample(min(n, candidate.sum().item()), feats=c_alpha)
            candidate_list = candidate.nonzero(as_tuple=True)[0]
            selected_idxs = candidate_list[candidate_idxs]
            print("candidate list", len(candidate_list), candidate_list)
        else:
            selected_idxs = np.array([], dtype=int)
            candidate_list = []

        print("selected_idxs", len(selected_idxs), selected_idxs)

        chosen_candidates = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_candidates[list(selected_idxs)] = True
        chosen_randoms = np.zeros(len(idxs_unlabeled), dtype=bool)

        all_candidates = np.zeros(len(idxs_unlabeled), dtype=bool)
        all_candidates[list(candidate_list)] = True

        if len(selected_idxs) < n:

            remained = n - len(selected_idxs)
            selected = np.zeros(len(idxs_unlabeled), dtype=bool)
            selected[selected_idxs] = True
            randoms = np.random.choice(np.where(selected == 0)[0], remained, replace=False)
            chosen = np.concatenate([selected_idxs, randoms])

            print('picked %d samples from RandomSampling.' % (remained))
            chosen_randoms[list(randoms)] = True
        else:
            chosen = selected_idxs

        # return np.array(selected_idxs), ulb_embedding, pred_1, ulb_probs, u_selected_idxs, idxs_unlabeled[candidate]
        query_idx = idxs_unlabeled[chosen]
        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[list(chosen)] = True

        correct = preds[diff:] == y[diff:]

        # return np.array(selected_idxs), ulb_embedding, pred_1, ulb_probs, u_selected_idxs, idxs_unlabeled[candidate]

        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), list(chosen_candidates), list(chosen_randoms), y[diff:], preds[diff:], correct, candidate.cpu().numpy(), query_idx],
            index=["id", "chosen", "chosen candidate", "chosen random", "label", "pred", "correct", "candidate", "query_idx"]).T
        return query_idx, query_df

    def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, Y, grads, clf):

        unlabeled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)

        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
        pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

        if self.alpha_closed_form_approx:
            alpha_cap /= math.sqrt(embedding_size)
            grads = grads.cuda(self.gpu)

        for i in range(self.args.num_classes):
            emb = lb_embedding[Y == i]
            if emb.size(0) == 0:
                emb = lb_embedding
            anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

            if self.alpha_closed_form_approx:
                embed_i, ulb_embed = anchor_i.cuda(self.gpu), ulb_embedding.cuda(self.gpu)
                alpha = self.calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)

                embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i
                out = clf(embedding_mix, only_fc=True)
                out = out.detach().cpu()
                alpha = alpha.cpu()

                pc = out.argmax(dim=1) != pred_1
            else:
                alpha = self.generate_alpha(unlabeled_size, embedding_size, alpha_cap)
                if self.alpha_opt:
                    alpha, pc = self.learn_alpha(ulb_embedding, pred_1, anchor_i, alpha, alpha_cap)
                else:
                    embedding_mix = (1 - alpha) * ulb_embedding + alpha * anchor_i
                    out = clf(embedding_mix.cuda(self.gpu), only_fc=True)
                    out = out.detach().cpu()

                    pc = out.argmax(dim=1) != pred_1

            torch.cuda.empty_cache()

            alpha[~pc] = 1.
            pred_change[pc] = True
            is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
            min_alphas[is_min] = alpha[is_min]
        return pred_change, min_alphas

    def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
        z = (lb_embedding - ulb_embedding)  # * ulb_grads
        alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (
                z + 1e-8)

        return alpha

    def sample(self, n, feats):
        feats = feats.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(feats)

        cluster_idxs = cluster_learner.predict(feats)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (feats - centers) ** 2
        dis = dis.sum(axis=1)
        return np.array(
            [np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
             (cluster_idxs == i).sum() > 0])

    def retrieve_anchor(self, embeddings, count):
        return embeddings.mean(dim=0).view(1, -1).repeat(count, 1)

    def generate_alpha(self, size, embedding_size, alpha_cap):
        alpha = torch.normal(
            mean=alpha_cap / 2.0,
            std=alpha_cap / 2.0,
            size=(size, embedding_size))

        alpha[torch.isnan(alpha)] = 1

        return self.clamp_alpha(alpha, alpha_cap)

    def clamp_alpha(self, alpha, alpha_cap):
        return torch.clamp(alpha, min=1e-8, max=alpha_cap)

    def learn_alpha(self, org_embed, labels, anchor_embed, alpha, alpha_cap, clf):
        labels = labels.cuda(self.gpu)
        min_alpha = torch.ones(alpha.size(), dtype=torch.float)
        pred_changed = torch.zeros(labels.size(0), dtype=torch.bool)

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        clf.eval()

        for i in range(self.alpha_learning_iters):
            tot_nrm, tot_loss, tot_clf_loss = 0., 0., 0.
            for b in range(math.ceil(float(alpha.size(0)) / self.alpha_learn_batch_size)):
                clf.zero_grad()
                start_idx = b * self.alpha_learn_batch_size
                end_idx = min((b + 1) * self.alpha_learn_batch_size, alpha.size(0))

                l = alpha[start_idx:end_idx]
                l = torch.autograd.Variable(l.cuda(self.gpu), requires_grad=True)
                opt = torch.optim.Adam([l], lr=self.alpha_learning_rate / (
                    1. if i < self.alpha_learning_iters * 2 / 3 else 10.))
                e = org_embed[start_idx:end_idx].cuda(self.gpu)
                c_e = anchor_embed[start_idx:end_idx].cuda(self.gpu)
                embedding_mix = (1 - l) * e + l * c_e

                out = clf(embedding_mix, only_fc=True)

                label_change = out.argmax(dim=1) != labels[start_idx:end_idx]

                tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool).cuda(self.gpu)
                tmp_pc[start_idx:end_idx] = label_change
                pred_changed[start_idx:end_idx] += tmp_pc[start_idx:end_idx].detach().cpu()

                tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (
                        l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).cuda(self.gpu))
                min_alpha[tmp_pc] = l[tmp_pc[start_idx:end_idx]].detach().cpu()

                clf_loss = loss_func(out, labels[start_idx:end_idx].cuda(self.gpu))

                l2_nrm = torch.norm(l, dim=1)

                clf_loss *= -1

                loss = self.alpha_clf_coef * clf_loss + self.alpha_l2_coef * l2_nrm
                loss.sum().backward(retain_graph=True)
                opt.step()

                l = self.clamp_alpha(l, alpha_cap)

                alpha[start_idx:end_idx] = l.detach().cpu()

                tot_clf_loss += clf_loss.mean().item() * l.size(0)
                tot_loss += loss.mean().item() * l.size(0)
                tot_nrm += l2_nrm.mean().item() * l.size(0)

                del l, e, c_e, embedding_mix
                torch.cuda.empty_cache()

        count = pred_changed.sum().item()
        return min_alpha.cpu(), pred_changed.cpu()
