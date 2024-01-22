from __future__ import division, absolute_import
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchreid.utils.tensortools import masked_mean


class PartSoftTripletLoss(nn.Module):
    def __init__(self, margin=0.3, epsilon=1e-16, writer=None):
        super(PartSoftTripletLoss, self).__init__()
        self.margin = margin
        self.writer = writer
        self.batch_debug = False
        self.imgs = None
        self.masks = None
        self.epsilon = epsilon
        self.enhence_head = 0.5

    def forward(self, part_based_embeddings, labels, parts_visibility=None):

        # Compute pairwise distance matrix for each part
        part_based_pairwise_dist = self._part_based_pairwise_distance_matrix(part_based_embeddings.transpose(1, 0),
                                                                             squared=False)

        if parts_visibility is not None:
            parts_visibility = parts_visibility.t()
            valid_part_based_pairwise_dist_mask = parts_visibility.unsqueeze(1) * parts_visibility.unsqueeze(2)
            if valid_part_based_pairwise_dist_mask.dtype is not torch.bool:
                valid_part_based_pairwise_dist_mask = torch.sqrt(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist_mask = None

        pairwise_dist = self._combine_part_based_dist_matrices(part_based_pairwise_dist,
                                                               valid_part_based_pairwise_dist_mask, labels)

        return self._hard_mine_triplet_loss(pairwise_dist, labels, self.margin)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        if valid_part_based_pairwise_dist_mask is not None:
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
            pairwise_dist = masked_mean(part_based_pairwise_dist, valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist = part_based_pairwise_dist
            pairwise_dist = valid_part_based_pairwise_dist.mean(0) * (5 / 5 + self.enhence_head)

        return pairwise_dist

    def _part_based_pairwise_distance_matrix(self, embeddings, squared=False):
        """
        embeddings.shape = (K, N, C)
        ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
        """
        # distances_head
        head_embeddings = embeddings[0:1]
        dot_product0 = torch.matmul(head_embeddings, head_embeddings.transpose(2, 1))
        square_sum0 = dot_product0.diagonal(dim1=1, dim2=2)
        distances = square_sum0.unsqueeze(2) - 2 * dot_product0 + square_sum0.unsqueeze(1)
        distances_head = F.relu(distances)

        # distances_other
        embeddings = embeddings[1:]
        dot_product = torch.matmul(embeddings, embeddings.transpose(2, 1))
        square_sum = dot_product.diagonal(dim1=1, dim2=2)
        distances = square_sum.unsqueeze(2) - 2 * dot_product + square_sum.unsqueeze(1)
        distances = F.relu(distances)

        distances_all = distances + distances_head * (1 + self.enhence_head)
        if not squared:
            mask = torch.eq(distances, 0).float()
            distances = distances + mask * self.epsilon  # for numerical stability (infinite derivative of sqrt in 0)
            distances = torch.sqrt(distances)
            distances = distances * (1 - mask)

        return distances

    def _hard_mine_triplet_loss(self, batch_pairwise_dist, labels, margin):

        max_value = torch.finfo(batch_pairwise_dist.dtype).max

        valid_pairwise_dist_mask = (batch_pairwise_dist != float(-1))

        self.writer.update_invalid_pairwise_distances_count(batch_pairwise_dist)

        # Get the hardest positive pairs
        # invalid positive distance were set to -1 to
        mask_anchor_positive = self._get_anchor_positive_mask(labels).unsqueeze(0)
        mask_anchor_positive = mask_anchor_positive * valid_pairwise_dist_mask
        valid_positive_dist = batch_pairwise_dist * mask_anchor_positive.float() - (~mask_anchor_positive).float()
        hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=-1)  # [K, N]

        # Get the hardest negative pairs
        mask_anchor_negative = self._get_anchor_negative_mask(labels).unsqueeze(0)
        mask_anchor_negative = mask_anchor_negative * valid_pairwise_dist_mask
        valid_negative_dist = batch_pairwise_dist * mask_anchor_negative.float() + (
            ~mask_anchor_negative).float() * max_value
        hardest_negative_dist, _ = torch.min(valid_negative_dist, dim=-1)  # [K, N]

        # Hardest negative/positive with dist=float.max/-1 are invalid: no valid negative/positive found for this anchor
        # Do not generate triplet for such anchor
        valid_hardest_positive_dist_mask = hardest_positive_dist != -1
        valid_hardest_negative_dist_mask = hardest_negative_dist != max_value
        valid_triplets_mask = valid_hardest_positive_dist_mask * valid_hardest_negative_dist_mask  # [K, N]
        hardest_dist = torch.stack([hardest_positive_dist, hardest_negative_dist], 2)  # [K, N, 2]
        valid_hardest_dist = hardest_dist[valid_triplets_mask, :]  # [K*N, 2]

        if valid_hardest_dist.nelement() == 0:
            warnings.warn("CRITICAL WARNING: no valid triplets were generated for current batch")
            return None

        # Build valid triplets and compute triplet loss
        if self.margin > 0:
            triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.hard_margin_triplet_loss(margin,
                                                                                                       valid_hardest_dist,
                                                                                                       valid_triplets_mask)
        else:
            triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.soft_margin_triplet_loss(0.3,
                                                                                                       valid_hardest_dist,
                                                                                                       valid_triplets_mask)

        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def hard_margin_triplet_loss(self, margin, valid_hardest_dist, valid_triplets_mask):
        triplet_losses = F.relu(valid_hardest_dist[:, 0] - valid_hardest_dist[:, 1] + margin)
        triplet_loss = torch.mean(triplet_losses)
        trivial_triplets_ratio = (triplet_losses == 0.).sum() / triplet_losses.nelement()
        valid_triplets_ratio = valid_triplets_mask.sum() / valid_triplets_mask.nelement()
        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def soft_margin_triplet_loss(self, margin, valid_hardest_dist, valid_triplets_mask):
        triplet_losses = F.relu(valid_hardest_dist[:, 0] - valid_hardest_dist[:, 1] + margin)
        hard_margin_triplet_loss = torch.mean(triplet_losses)
        trivial_triplets_ratio = (triplet_losses == 0.).sum() / triplet_losses.nelement()
        valid_triplets_ratio = valid_triplets_mask.sum() / valid_triplets_mask.nelement()

        # valid_hardest_dist[:, 0] = hardest positive dist
        # valid_hardest_dist[:, 1] = hardest negative dist
        y = valid_hardest_dist[:, 0].new().resize_as_(valid_hardest_dist[:, 0]).fill_(1)
        soft_margin_triplet_loss = F.soft_margin_loss(valid_hardest_dist[:, 1] - valid_hardest_dist[:, 0], y)
        if soft_margin_triplet_loss == float('Inf'):
            print("soft_margin_triplet_loss = inf")
            return hard_margin_triplet_loss, trivial_triplets_ratio, valid_triplets_ratio
        return soft_margin_triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    @staticmethod
    def _get_anchor_positive_mask(labels):
        """
        To be a valid positive pair (a,p) :
            - a and p are different embeddings
            - a and p have the same label
        """
        indices_equal_mask = torch.eye(labels.shape[0], dtype=torch.bool,
                                       device=(labels.get_device() if labels.is_cuda else None))
        indices_not_equal_mask = ~indices_equal_mask

        # Check if labels[i] == labels[j]
        labels_equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

        mask_anchor_positive = indices_not_equal_mask * labels_equal_mask

        return mask_anchor_positive

    @staticmethod
    def _get_anchor_negative_mask(labels):
        """
        To be a valid negative pair (a,n) :
            - a and n have different labels (and therefore are different embeddings)
        """

        # Check if labels[i] != labels[k]
        labels_not_equal_mask = torch.ne(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

        return labels_not_equal_mask
