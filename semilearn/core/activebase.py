import torch


class ActiveBase:
    def __init__(self, gpu):
        self.idxs_lb = None
        self.n_pool = None
        self.gpu = gpu

    def query(self, n, clf, data_loaders):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def get_x_y_embs_logits_probs_preds(self, clf, data_loader):
        """
        get embedding function
        """
        clf.eval()

        all_true = []
        all_pred = []
        all_embs = []
        all_logits = []
        all_x = []
        all_probs = []
        with torch.no_grad():
            for data in data_loader:
                x = data["x_lb"]
                y = data["y_lb"]

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                out = clf(x)
                emb, logits = out["feat"], out["logits"]
                probs = torch.softmax(logits, dim=-1)

                all_true.extend(y.cpu().tolist())
                all_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                all_logits.append(logits.cpu().numpy())
                all_embs.append(emb.cpu().numpy())
                all_x.append(x.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        all_logits = np.concatenate(all_logits)
        all_embs = np.concatenate(all_embs)
        all_x = np.concatenate(all_x)
        all_probs = np.concatenate(all_probs)

        clf.train()

        return all_x, all_true, all_embs, all_logits, all_probs, all_pred