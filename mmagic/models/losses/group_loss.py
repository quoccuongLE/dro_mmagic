# Copyright (c) Quoc Cuong LE. All rights reserved.
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmagic.registry import MODELS


def parsing_group_metadata(group_mapping: Union[str, dict]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """_summary_

    Args:
        group_mapping (Union[str, dict]): _description_

    Returns:
        Tuple[Dict[int, int], Dict[int, int]]: _description_
    """
    if isinstance(group_mapping, str):
        with open(group_mapping, "r") as f:
            group_mapping_metadata = json.load(f)
    elif isinstance(group_mapping, dict):
        group_mapping_metadata = group_mapping
    else:
        raise TypeError("Only accept dictionary and string path to JSON file !")

    group_counts, group_map = {}, {}
    for cls_meta in group_mapping_metadata.values():
        if cls_meta["group_id"] not in group_counts.keys():
            group_counts[cls_meta["group_id"]] = 0
        group_counts[cls_meta["group_id"]] += cls_meta["count"]
        group_map[cls_meta["class_id"]] = cls_meta["group_id"]

    group_counts = dict(sorted(group_counts.items()))

    return group_map, group_counts


def _averaging_by_group(
        ndarray: torch.Tensor,
        group_count: torch.Tensor,
        group_ids: torch.Tensor,
        group_id_mat: torch.Tensor) -> torch.Tensor:
    """Computes observed counts and mean loss for each group
    Args:
            losses (torch.Tensor): Loss to compute
            group_count (List[int]): Counts in each group
            group_ids (list[int]): List of group ids
            group_id_mat (torch.Tensor): Matrix containing group ID pairwise

    Returns:
            torch.Tensor: Group loss

    """
    # Compute observed counts and mean loss for each group
    n_groups = len(group_ids)
    # [nb_groups, ...] add new dimension (groups)
    arange_arr = torch.stack([group_id*torch.ones_like(group_id_mat, device=ndarray.device) for group_id in group_ids])
    # [nb_groups, ...] = [nb_groups, ...] == [...]
    group_mask = (arange_arr == group_id_mat).float()
    # [nb_groups x 1]
    group_denominator = group_count + (group_count==0).float()
    # [nb_groups x len(arr.view(-1))] -> two dimensions only
    group_nominator = group_mask.view(n_groups, -1) * ndarray.view(-1)
    # [nb_groups x 1]
    group_val = group_nominator.sum(1) / group_denominator
    # del arange_arr, group_nominator, group_denominator
    return group_val


@MODELS.register_module()
class GroupLoss(nn.Module):
    """Group loss.
    Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization

    This implementation is forked from the Github repo https://github.com/kohpangwei/group_DRO
    Paper: https://arxiv.org/abs/1911.08731
    Authors: Shiori Sagawa*, Pang Wei Koh*, Tatsunori Hashimoto, and Percy Liang

    """
    group_counts: Dict[int, int]
    group_map: Dict[int, int]

    def __init__(self,
                group_mapping: Union[str, dict],
                group_counts: Optional[List[int]] = None,
                group_map: Optional[List[int]] = None,
                is_robust: bool = True,
                alpha: float = 0.02,
                gamma: float = 0.1,
                adj: float = 2.,
                min_var_weight: float = 0.0,
                step_size: float = 0.01,
                normalize_loss: bool = False,
                btl: bool = False,
                reduction: str = 'none',
                loss_weight: float = 1.0,
                avg_factor: bool = False,
                *args, **kwargs):
        """Group loss for Distribution Robust Optimization (DRO)

        Args:
            group_counts (list[int]): Number of samples per group.
            criterion (str, optional): Base Torch loss. Defaults to nn.CrossEntropyLoss(reduction='none').
            expert (bool, optional): Whether to set the loss in expert mode. Defaults to False.
            alpha (float, optional): Hyperparameter ???. Defaults to 0.02.
            gamma (float, optional): Hyperparameter for the exponential loss. Defaults to 0.1.
            adj (np.ndarray, optional): Adjustment hyperparameter `model capacity constant` C in eq. 5 in the paper. Defaults to None.
            min_var_weight (float, optional): Hyperparameter ???. Defaults to 0.0.
            step_size (float, optional): Hyperparamter \eta_{q} in the Algorithm 1. Defaults to 0.01.
            normalize_loss (bool, optional): Flag indicating to normalize loss. Defaults to False.
            btl (bool, optional): Flag for an alternative. Defaults to False.
            reduction (str, optional): Argument for Torch loss. Defaults to 'none'.
        """
        super().__init__(*args, **kwargs)

        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.avg_factor = avg_factor

        group_map, group_counts = parsing_group_metadata(group_mapping=group_mapping)
        self.group_map = group_map
        self.group_ids = torch.IntTensor(list(set(group_map.values())))

        self._group_counts = torch.IntTensor(list(group_counts.values()))
        self.n_groups = len(group_counts)
        self.group_frac = torch.FloatTensor(list(group_counts.values())) / sum(list(group_counts.values()))
        self.group_str = None # Group names in the dataset

        if adj is not None:
            if isinstance(adj, float):
                self.adj = torch.ones(self.n_groups, dtype=torch.float) * adj
            else:
                raise ValueError(f'adj type {type(adj)} unsupported !')
        else:
            self.adj = torch.zeros(self.n_groups, dtype=torch.float)

        if self.is_robust:
            assert alpha, 'Alpha must be specified!'

        # Quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups) / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte()

    def loss(self,
        group_id_mat: torch.Tensor,
        group_ids: List[int],
        group_count: Union[List[int], torch.Tensor],
        per_sample_losses: Optional[torch.Tensor] = None,
        *args, **kwargs) -> torch.Tensor:
        """This function computes per-sample and per-group losses

        Args:
            yhat (torch.Tensor): Predictions including neither class indices in the range
                [0,C), where C is the number of classes, or probabilities for each
                class.
            y (torch.Tensor): One-hot ground truth vector
            group_ids (list[int]): List of group ids.

        Returns:
            (torch.Tensor): Final loss
        """

        # Group loss computation
        _loss_per_group = {}
        for idx, group_id in enumerate(group_ids):
            assert idx == group_id, "This implementation only accepts group IDs, which also are indexes, e.g. from 0 to N"
            _loss_per_group[group_id] = per_sample_losses[group_id_mat == group_id, ...]

        # compute overall loss
        if self.is_robust:
            loss_per_group = [loss.mean() / count for loss, count in zip(_loss_per_group.values(), group_count)]
            loss_per_group = torch.stack(loss_per_group)
            # Update historical losses
            self.update_exp_avg_loss(loss_per_group.to(device="cpu"), group_count.to(device="cpu"))
            if self.btl:
                actual_loss, weights = self.compute_robust_loss_btl(loss_per_group)
            else:
                actual_loss, weights = self.compute_robust_loss(loss_per_group)

        else:
            weights = None
            if self.avg_factor:
                actual_loss = per_sample_losses.mean()
            else:
                actual_loss = per_sample_losses.sum()
            with torch.no_grad():
                loss_per_group = torch.stack(list(_loss_per_group.values()))
                loss_per_group = loss_per_group / self._group_counts

        return actual_loss * self.loss_weight

    def compute_robust_loss(self, group_loss: torch.Tensor):
        """Computes final loss giving loss per groups
        Args:
            group_loss (torch.Tensor): Loss compounding individual group

        """
        if torch.all(self.adj > 0):
            adjusted_loss = group_loss + self.adj.to(group_loss.device) / torch.sqrt(self._group_counts.to(group_loss.device))
        if self.normalize_loss:
            adjusted_loss /= adjusted_loss.sum()
        self.adv_probs *= torch.exp(self.step_size * adjusted_loss.data.to(device="cpu"))
        self.adv_probs /= (self.adv_probs.sum())

        # Multiplication matrix
        # https://stackoverflow.com/questions/27385633/what-is-the-symbol-for-in-python
        robust_loss = group_loss @ self.adv_probs.to(device=group_loss.device)
        return robust_loss, self.adv_probs

    def compute_group_avg(self, losses: torch.Tensor, group_ids: List[int]) -> Tuple[torch.Tensor, List[float]]:
        """Computes observed counts and mean loss for each group
        Args:
            losses (torch.Tensor): Loss to compute.
            group_ids (list[int]): List of group ids.

        Returns:
            torch.Tensor: Group loss.
            list[float]: Group count.

        """
        group_map = (group_ids == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss: torch.Tensor, group_count: torch.Tensor):
        """Update the exp avg loss

        Args:
            group_loss (torch.Tensor): Group loss.
            group_count (torch.Tensor): group count.
        """
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def compute_robust_loss_btl(self, group_loss: torch.Tensor):
        adjusted_loss = self.exp_avg_loss + self.adj.to(group_loss.device) / torch.sqrt(self._group_counts.to(group_loss.device))
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss: torch.Tensor, ref_loss: torch.Tensor):
        sorted_loss, sorted_idx = ref_loss.sort(descending=True)
        sorted_frac = self.group_frac[sorted_idx].to(group_loss.device)

        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        weights = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def forward(self,
        precomputed_loss: Optional[torch.Tensor] = None,
        group_count: Optional[List[int]] = None,
        group_id_mat: Optional[List[int]] = None,
        group_ids: Optional[List[int]] = None,
        is_training: bool = False,
        device: str = "cpu",
        *args, **kwargs) -> Tuple[torch.Tensor, dict]:
        """Group loss

        Args:
            precomputed_loss (torch.Tensor, optional): Precomputed loss TODO: more details
            group_count (List[int], optional): Number of elements per group. Defaults to None.
            group_ids (List[int], optional): TODO: . Defaults to None.
            is_training (bool, optional): Flag indicating if the forwarding is for
                training. Defaults to False.
            device (str, optional): Device to place tensors while calculating, Defaults to 'cpu'.

        Returns:
            Tuple[torch.Tensor, dict]: Loss and log
        """

        loss = self.loss(
                group_id_mat=group_id_mat,
                group_ids=group_ids,
                is_training=is_training,
                group_count=torch.Tensor(group_count).to(device),
                per_sample_losses=precomputed_loss,
                *args,
                **kwargs,
            )

        return loss
