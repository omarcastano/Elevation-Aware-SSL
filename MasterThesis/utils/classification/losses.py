import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NTXentLoss

    Args:
        tau: The temperature parameter.
    """

    def __init__(self, tau: float = 0.5, eps: float = 1e-8):

        super(NTXentLoss, self).__init__()
        self.tau = tau
        self.eps = eps

    def _get_pos_mask(self, bs):

        """
        Get positive mask

        Argumetns:
        ---------
            bs: batch size
        """

        zeros = torch.zeros((bs, bs), dtype=torch.uint8)
        eye = torch.eye(bs, dtype=torch.uint8)
        pos_mask = torch.cat(
            [
                torch.cat([zeros, eye], dim=0),
                torch.cat([eye, zeros], dim=0),
            ],
            dim=1,
        )

        return pos_mask.type(torch.bool)

    def _get_neg_mask(self, bs):

        """
        Get negative mask

        Argumetns:
        ---------
            bs: batch size
        """

        diag = np.eye(2 * bs)
        l1 = np.eye((2 * bs), 2 * bs, k=-bs)
        l2 = np.eye((2 * bs), 2 * bs, k=bs)

        mask = torch.from_numpy((diag + l1 + l2))
        neg_mask = 1 - mask

        return neg_mask.type(torch.bool)

    def forward(self, zi, zj):
        """
        input: {'zi': out_feature_1, 'zj': out_feature_2}
        target: one_hot lbl_prob_mat
        """

        zi, zj = F.normalize(zi, dim=1), F.normalize(zj, dim=1)
        bs = zi.shape[0]

        pos_mask = self._get_pos_mask(bs).to(zi.device)
        neg_mask = self._get_neg_mask(bs).to(zi.device)

        z_all = torch.cat([zi, zj], dim=0)  # input1,input2: z_i,z_j

        # [2*bs, 2*bs] -  pairwise similarity
        sim_mat = torch.exp(torch.mm(z_all, z_all.t().contiguous()) / self.tau)

        sim_pos = sim_mat.masked_select(pos_mask).view(2 * bs).clone()
        sim_neg = sim_mat.masked_select(neg_mask).view(2 * bs, -1)

        loss = (-torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()

        return loss
