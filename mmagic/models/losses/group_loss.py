# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses import binary_cross_entropy, cross_entropy, sigmoid_focal_loss

from mmyolo.registry import MODELS


def _averaging_by_group_legacy(arr: torch.Tensor, n_groups: int, group_ids: List[int]) -> Tuple[torch.Tensor, List[float]]:
  """Computes observed counts and mean loss for each group
  Args:
      arr (torch.Tensor): N-d array to compute over groups.
      group_ids (list[int]): List of group ids.

  Returns:
      torch.Tensor: Arrays by group.
      list[float]: Group count.

  """
  # Compute observed counts and mean loss for each group
  # [nb_groups x batch_size or n_samples]
  group_map = (torch.Tensor(group_ids) == torch.arange(n_groups).unsqueeze(1).long().cuda()).float()
  # [nb_groups x 1]
  group_count = group_map.sum(1)
  # [nb_groups x 1]
  group_denom = group_count + (group_count==0).float() # avoid NANs
  # [nb_groups] = [nb_groups x batch_size or n_samples] x [batch_size or n_samples]
  group_val = (group_map @ arr.view(-1))/group_denom
  return group_val, group_count


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
  # group_mask = (torch.stack(n_groups*[group_id_mat]) == arange_arr).cuda().float()
  group_mask = (arange_arr == group_id_mat).float()
  # [nb_groups x 1]
  # group_denominator = (group_count + (group_count==0).float()).cuda()
  group_denominator = group_count + (group_count==0).float()
  # [nb_groups x len(arr.view(-1))] -> two dimensions only
  group_nominator = group_mask.view(n_groups, -1) * ndarray.view(-1)
  # [nb_groups x 1]
  group_val = group_nominator.sum(1) / group_denominator
  return group_val


def _distrib_group_mapping(target_distrib: List[float], source: List[float]) -> List[float] :
  """Maps a target distribution and a source histogram.

  Args:
    target_distrib (list[float]): Target distribution e.g. [.8,.15,.05].
    source (list[float], optional): Source distribution
      e.g. [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50].

  Returns:
    list[float]: source's size list in which each index is a class. 
                the value taken by the list at index i corresponds 
                to the group to which class i belongs.
    list[int]: target histogram.
  """
  cum_sum = np.cumsum(source)
  n_tot = cum_sum[-1]
  target = np.array(target_distrib)*n_tot
  target = np.round(target).astype(np.int)
  target[-1] = n_tot - np.cumsum(target)[-2]
  
  cum_sum_target = np.cumsum(target)
  ret = [0]*len(source)
  k,pos = 0,0
  while pos<len(cum_sum) :
    if cum_sum_target[k]<cum_sum[pos] :
      k+=1
      ret[pos]=k
    else :
      ret[pos]=k
    pos+=1
  
  assert np.cumsum(target)[-1] == np.cumsum(source)[-1]

  return list(ret), list(target)


def _group_parsing(coords: Dict,
    output_shape: List[int],
    augmented_mat: Optional[torch.Tensor]=None):
  output = torch.zeros(output_shape)
  for grp_id, group_coord in coords.items():
    output[group_coord] = grp_id
  if augmented_mat is not None:
    # TODO: To be reviewed
    # Masked entities of augmented_mat are -1
    output += augmented_mat

  return output


def _get_model_stats(model: nn.Module, args: Any, stats_dict: dict) -> dict:
  """Update statistics with model info

  Args:
    model (torch.Module): Model to extract extra info
    args (Any): Arguments to extract model's info
    stats_dict (dict): Stats dict to update

  Returns:
    dict: Updated stats dict
  """
  model_norm_sq = sum(torch.norm(param) ** 2 for param in model.parameters())
  stats_dict['model_norm_sq'] = model_norm_sq.item()
  stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
  return stats_dict


@MODELS.register_module()
class GroupLoss(nn.Module):
  """Group loss.
  Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization

  This implementation is forked from the Github repo https://github.com/kohpangwei/group_DRO
  Paper: https://arxiv.org/abs/1911.08731
  Authors: Shiori Sagawa*, Pang Wei Koh*, Tatsunori Hashimoto, and Percy Liang

  """
  def __init__(self,
      group_counts: List[int],
      group_map: Optional[List[int]] = None,
      augmented_group: bool = False,
      is_robust: bool = True,
      mode: str = 'single',
      criterion: str = 'CrossEntropy',
      target_distribution: List[float] = [.8, .15, .05],
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
      accuracy_metric_builtin: bool = True,
      *args, **kwargs):
    """Group loss for Distribution Robust Optimization (DRO)

    Args:
        group_counts (list[int]): Number of samples per group.
        criterion (str, optional): Base Torch loss. Defaults to nn.CrossEntropyLoss(reduction='none').
        expert (bool, optional): Whether to set the loss in expert mode. Defaults to False.
        target_distribution : Group distribution in expert mode. Defaults to [80%, 15%, 5%]
        alpha (float, optional): Hyperparameter ???. Defaults to 0.02.
        gamma (float, optional): Hyperparameter for the exponential loss. Defaults to 0.1.
        adj (np.ndarray, optional): Adjustment hyperparameter `model capacity constant` C in eq. 5 in the paper. Defaults to None.
        min_var_weight (float, optional): Hyperparameter ???. Defaults to 0.0.
        step_size (float, optional): Hyperparamter \eta_{q} in the Algorithm 1. Defaults to 0.01.
        normalize_loss (bool, optional): Flag indicating to normalize loss. Defaults to False.
        btl (bool, optional): Flag for an alternative. Defaults to False.
        reduction (str, optional): Argument for Torch loss. Defaults to 'none'.
    """
    super().__init__()

    # assert np.cumsum(target_distribution)[-1] == 1. , "The sum of probability must be equal to 1.0"
    assert mode in ('single', 'group', 'descending_distribution')

    if criterion == 'Focal': #TODO fix implementation
      self.criterion = sigmoid_focal_loss
    elif criterion == 'CrossEntropy':
      self.criterion = cross_entropy
    elif criterion == 'BinaryCrossEntropy':
      self.criterion = binary_cross_entropy
    else: 
      raise(Exception('criterion can be either Focal or CrossEntropy'))

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
    self.mode = mode
    self.accuracy_builtin = accuracy_metric_builtin

    if self.mode == 'descending_distribution': 
      # self.target_distribution = target_distribution
      # self.group_map, group_counts = _distrib_group_mapping(target_distribution, group_counts)
      raise NotImplementedError

    elif self.mode == 'group':
      self.group_map = group_map
      self.group_ids = torch.IntTensor(list(set(group_map)))
    else:
      self.group_map = None

    self.augmented_group = augmented_group
    self._group_counts = torch.IntTensor(group_counts.values())
    self.n_groups = len(group_counts) + 1 if augmented_group else len(group_counts)
    self.group_frac = torch.FloatTensor(group_counts.values()) / sum(group_counts.values())
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

    # quantities maintained throughout training
    self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
    self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
    self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

    self.reset_stats()

  def loss(self,
      pred: torch.Tensor,
      target: torch.Tensor,
      group_id_mat: torch.Tensor,
      group_ids: List[int],
      group_count: Union[List[int], torch.Tensor],
      per_sample_losses: Optional[torch.Tensor] = None,
      is_training: Optional[bool] = False,
      *args, **kwargs) -> torch.Tensor:
    """
    This function computes per-sample and per-group losses

    Args:
      yhat (torch.Tensor): Predictions including neither class indices in the range
        [0,C), where C is the number of classes, or probabilities for each
        class.
      y (torch.Tensor): One-hot ground truth vector
      group_ids (list[int]): List of group ids.

    Returns:
      (torch.Tensor): Final loss
    """
    # Loss computation
    if per_sample_losses is None:
      assert pred.size(0) == target.size(0) and target.numel() > 0
      if self.criterion == binary_cross_entropy:
        per_sample_losses = self.criterion(pred, target, reduction='none')
      else:
        per_sample_losses = self.criterion(pred, target, reduction=self.reduction)

    # Group loss computation
    group_loss = _averaging_by_group(
      ndarray=per_sample_losses,
      group_count=group_count,
      group_ids=group_ids,
      group_id_mat=group_id_mat)

    if self.accuracy_builtin:
      # Group accuracy computation
      if target.dim() == 1:
        accuracy = (torch.argmax(pred, dim=1) == target).float()
        reduced_group_id_mat = group_id_mat
      else:
        # One-hot label vector y
        accuracy = (torch.argmax(pred, dim=1) == torch.argmax(target, dim=1)).float()
        reduced_group_id_mat = group_id_mat[:, 0]
      with torch.no_grad():
        group_accuracy = _averaging_by_group(
          ndarray=accuracy,
          group_count=group_count,
          group_ids=group_ids,
          group_id_mat=reduced_group_id_mat)
    else:
      group_accuracy = None

    # Update historical losses
    self.update_exp_avg_loss(group_loss, group_count)

    # compute overall loss
    if not self.is_robust:
      weights = None
      if self.avg_factor:
        actual_loss = per_sample_losses.mean()
      else:
        actual_loss = per_sample_losses.sum()
    else:
      if self.btl:
        actual_loss, weights = self.compute_robust_loss_btl(group_loss)
      else:
        actual_loss, weights = self.compute_robust_loss(group_loss)
      if not self.avg_factor:
        # TODO: Review this code. Only correct for mmcls, not mmdet. In mmdet,
        # avg_factor = True, meaning never run through this part
        # Adaptation to the implementation of loss in MMdetection
        # An averaging over the total number of samples (num_total_samples) outside of loss function
        # By default, avg_factor should be None to comply with losses object in MMDetection
        actual_loss *= per_sample_losses.shape[0]

    # update stats
    self.update_stats(
        actual_loss=actual_loss,
        group_loss=group_loss,
        group_count=group_count,
        group_acc=group_accuracy,
        weights=weights)

    return actual_loss*self.loss_weight

  def compute_robust_loss(self, group_loss: torch.Tensor):
    """Computes final loss giving loss per groups
    Args:
        group_loss (torch.Tensor): Loss compounding individual group

    """
    adjusted_loss = group_loss
    if torch.all(self.adj>0):
      adjusted_loss += self.adj.to(group_loss.device)/torch.sqrt(self._group_counts.to(group_loss.device))
    if self.normalize_loss:
      adjusted_loss = adjusted_loss/(adjusted_loss.sum())
    self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
    self.adv_probs = self.adv_probs/(self.adv_probs.sum())

    # Multiplication matrix
    # https://stackoverflow.com/questions/27385633/what-is-the-symbol-for-in-python
    robust_loss = group_loss @ self.adv_probs
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

  def update_exp_avg_loss(self, group_loss: torch.Tensor, group_count: List[int]):
    """Update the exp avg loss

    Args:
      group_loss (torch.Tensor): Group loss.
      group_count (list[float]): group count.
    """
    prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
    curr_weights = 1 - prev_weights
    self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
    self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

  @torch.no_grad()
  def update_stats(self,
      actual_loss: torch.Tensor,
      group_loss: torch.Tensor,
      group_count: torch.Tensor,
      group_acc: Optional[torch.Tensor] = None,
      weights: Optional[torch.Tensor] = None):
    """Keeps useful stats for loss computation updated through training phases

    Args:
      actual_loss (torch.Tensor): The final loss for gradient decent step
      group_loss (torch.Tensor): Precomputed loss wrt group
      group_acc (torch.Tensor): Accuracy in group
      group_count (list[float]): Group count
      weights (torch.Tensor, optional): Weight of groups. Defaults to None.
    """
    # avg group loss
    denom = self.processed_data_counts + group_count
    denom += (denom == 0).float()
    prev_weight = self.processed_data_counts/denom
    curr_weight = group_count/denom
    self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

    # avg group acc
    if self.accuracy_builtin:
      self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

    # batch-wise average actual loss
    denom = self.batch_count + 1
    self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

    # counts
    self.processed_data_counts += group_count
    if self.is_robust:
      self.update_data_counts += group_count*((weights>0).float())
      self.update_batch_counts += ((group_count*weights)>0).float()
    else:
      self.update_data_counts += group_count
      self.update_batch_counts += (group_count>0).float()
    self.batch_count+=1

    # avg per-sample quantities
    group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
    self.avg_per_sample_loss = group_frac @ self.avg_group_loss
    if self.accuracy_builtin:
      self.avg_acc = group_frac @ self.avg_group_acc

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

  @torch.no_grad()
  def get_stats(self, model: Optional[nn.Module]=None, args: Optional[Any]=None) -> Dict:
    """Get statistics of groups during training for monitoring

    Args:
        model (Optional[nn.Module], optional): Model to extract extra info
          beyond defaut stats of groups. Defaults to None.
        args (Optional[Any], optional): Arguments to extract model's info.
          Defaults to None.

    Returns:
        Dict[Any]: Dict containing stats
    """
    stats_dict = {}
    for idx in range(self.n_groups):
      stats_dict[f'loss_group/{idx}'] = torch.Tensor(np.array(self.avg_group_loss[idx].item()))
      stats_dict[f'exp_loss_group/{idx}'] = torch.Tensor(np.array(self.exp_avg_loss[idx].item()))
      stats_dict[f'acc_group/{idx}'] = torch.Tensor(np.array(self.avg_group_acc[idx].item()))
      stats_dict[f'processed_data_count_group/{idx}'] = torch.Tensor(np.array(self.processed_data_counts[idx].item()))
      stats_dict[f'update_data_count_group/{idx}'] = torch.Tensor(np.array(self.update_data_counts[idx].item()))
      stats_dict[f'update_batch_count_group/{idx}'] = torch.Tensor(np.array(self.update_batch_counts[idx].item()))

    stats_dict['avg_actual_loss'] = torch.Tensor(np.array(self.avg_actual_loss.item()))
    stats_dict['avg_per_sample_loss'] = torch.Tensor(np.array(self.avg_per_sample_loss.item()))
    stats_dict['avg_acc'] = torch.Tensor(np.array(self.avg_acc.item()))

    # Model stats
    if model is not None:
      assert args is not None
      stats_dict = _get_model_stats(model, args, stats_dict)

    return stats_dict

  def reset_stats(self):
    self.processed_data_counts = torch.zeros(self.n_groups).cuda()
    self.update_data_counts = torch.zeros(self.n_groups).cuda()
    self.update_batch_counts = torch.zeros(self.n_groups).cuda()
    self.avg_group_loss = torch.zeros(self.n_groups).cuda()
    self.avg_group_acc = torch.zeros(self.n_groups).cuda()
    self.avg_per_sample_loss = torch.zeros(1).cuda()
    self.avg_actual_loss = torch.zeros(1).cuda()
    self.avg_acc = torch.zeros(1).cuda()
    self.batch_count = torch.zeros(1).cuda()

  def forward(self,
      pred: torch.Tensor,
      target: torch.Tensor,
      precomputed_loss: Optional[torch.Tensor] = None,
      group_id_coords: Optional[Union[dict, torch.Tensor]] = None,
      group_count: Optional[List[int]] = None,
      group_ids: Optional[List[int]] = None,
      augmented_group: Optional[torch.Tensor] = None,
      is_training: bool = False,
      *args, **kwargs) -> Tuple[torch.Tensor, dict]:
    """Group loss

    Args:
      pred (torch.Tensor): Prediction tensor
      target (torch.Tensor): Labels to compare with predicitions
      is_training (bool, optional): Flag indicating if the forwarding is for
        training. Defaults to False.

    Returns:
      Tuple[torch.Tensor, dict]: Loss and log
    """
    if isinstance(group_id_coords, dict):
      group_id_mat = _group_parsing(
        coords=group_id_coords,
        output_shape=target.shape,
        augmented_mat=augmented_group)
      if augmented_group is not None:
        group_count.insert(0, target.shape.numel() - sum(group_count))
        group_ids = [-1] + list(group_id_coords.keys())
    elif isinstance(group_id_coords, torch.Tensor):
      # group_ids = [-1] + list(set(self.group_ids))
      group_ids = self.group_ids.to(group_id_coords.device)
      if group_id_coords.shape[-1] == len(group_ids):
        # Is broadcastable ?
        group_mask = group_id_coords == group_ids
      else:
        group_mask = group_id_coords[..., None] == group_ids
      group_count = group_mask.int().sum(0)
      group_id_mat = group_id_coords
    else:
      raise TypeError(f'Type {type(group_id_coords)} is not supported!')

    if precomputed_loss is None:
      assert group_id_mat.shape == target.shape
    loss = self.loss(
        pred=pred,
        target=target,
        group_ids=group_ids,
        group_id_mat=group_id_mat,
        is_training=is_training,
        group_count=torch.Tensor(group_count).to(group_id_mat.device),
        per_sample_losses=precomputed_loss,
        *args, **kwargs)

    return loss
