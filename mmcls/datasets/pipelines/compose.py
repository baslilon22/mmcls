# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES
import torchvision
import torch

@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        # count = 0
        for t in self.transforms:
            # count += 1
            # if count == 2:
            #     img_normalized = (data['img'] - data['img'].min()) / (data['img'].max() - data['img'].min())  # 规范化为 [0, 1]
            #     img_scaled = img_normalized * 2 - 1  # 缩放为 [-1, 1]
            #     img_scaled = torch.from_numpy(img_scaled)
            #     img_scaled = img_scaled.transpose(0,2).transpose(1,2)
            #     torchvision.utils.save_image(img_scaled, 'before_transform.jpg', nrow=10)

                
            data = t(data)
            if data is None:
                return None
        # # 规范化和缩放图像
        # img_normalized = (data['img'] - data['img'].min()) / (data['img'].max() - data['img'].min())  # 规范化为 [0, 1]
        # img_scaled = img_normalized * 2 - 1  # 缩放为 [-1, 1]
        # torchvision.utils.save_image(img_scaled, 'after_transform.jpg', nrow=10)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string
