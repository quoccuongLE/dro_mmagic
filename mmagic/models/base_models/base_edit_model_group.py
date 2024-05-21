# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from .base_edit_model import BaseEditModel

from mmagic.registry import MODELS
from mmagic.structures import DataSample


@MODELS.register_module(name="BaseEditModelGroup")
class BaseEditModelGroup(BaseEditModel):
    """Base model for image and video editing.

    It must contain a generator that takes frames as inputs and outputs an
    interpolated frame. It also has a pixel-wise loss for training.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.

    Attributes:
        init_cfg (dict, optional): Initialization config dict.
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`. Default: None.
    """

    def __init__(
        self,
        generator: dict,
        pixel_loss: dict,
        group_loss: Optional[dict] = None,
        group_count: Optional[list[int]] = None,
        group_ids: Optional[list[int]] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
    ):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg, init_cfg, data_preprocessor)
        if group_loss:
            self.group_count = group_count
            self.group_ids = group_ids
            self.group_loss = MODELS.build(group_loss)
            self.pixel_loss.sample_wise = True
        else:
            self.group_loss = None

    def forward_train(
        self, inputs: torch.Tensor, data_samples: Optional[List[DataSample]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward training. Returns dict of losses of training.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        batch_gt_data = data_samples.gt_img

        loss = self.pixel_loss(feats, batch_gt_data)
        if self.group_loss:
            pairwise_group_labels = torch.Tensor(data_samples.group_id).to(device=loss.device)
            loss = self.group_loss(
                group_id_mat=pairwise_group_labels,
                precomputed_loss=loss,
                group_count=self.group_count,
                group_ids=self.group_ids,
            )

        return dict(loss=loss)
