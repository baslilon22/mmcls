# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from abc import ABCMeta, abstractmethod
from os import PathLike
from typing import List

import mmcv
import numpy as np
from torch.utils.data import Dataset
import random
from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy
from .pipelines import Compose
from typing import Final



def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(self,
                 data_prefix,
                 pipeline,
                 Batch_size=0,#这个是后来加上去的。
                 classes=None,
                 ann_file=None,
                 test_mode=False,
                 ):
        super(BaseDataset, self).__init__()
        self.data_prefix = expanduser(data_prefix)
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.ann_file = expanduser(ann_file)
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()
        self.origin_data_infos = self.data_infos.copy()
        
        #后来加上去的
        self.Batch_size = Batch_size
        self.ratio_NP = 0.2 #正负样本比
        if self.test_mode == False and ('negative' in self.CLASSES):
            self.current_iterative = 0
            self.negative_count = 0
            #计算数据集负样本的总量。
            for i in range(len(self.data_infos)):
                # 修改self.data_infos中的元素
                if int(self.data_infos[i]['gt_label']) == self.CLASSES.index('negative'):
                    self.negative_count += 1
            self.positive_count = len(self.data_infos) - self.negative_count #正样本数量，后面如果有小负样本池就需要进行修改。
            if self.negative_count / self.positive_count > self.ratio_NP:
                # 将gt_label等于0的项存储到一个新列表中
                selected_data = [data for data in self.data_infos if data['gt_label'] == self.CLASSES.index('negative')]
                self.data_infos = [data for data in self.data_infos if data['gt_label'] != self.CLASSES.index('negative')]
                # 从selected_data中随机选择一部分进行删除
                # num_to_delete = int(len(selected_data) * ((self.negative_count - self.positive_count) / self.negative_count))  # 删除一半的项，可以根据需求调整比例
                num_choice = int(self.positive_count * self.ratio_NP)#这里是选择保留这个数量的负样本
                items_choice = random.sample(selected_data, num_choice)
                # 从data_list中删除选定的项
                self.data_infos = items_choice + self.data_infos

            self.Reconstruct_Frequency =  int(len(self.data_infos) / self.Batch_size)

    @abstractmethod
    def load_annotations(self):
        pass

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return [int(self.data_infos[idx]['gt_label'])]

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        if self.test_mode == False and ('negative' in self.CLASSES):
            self.current_iterative += 1
            if self.current_iterative % self.Reconstruct_Frequency == 0 and self.negative_count / self.positive_count > self.ratio_NP:
                self.data_infos = self.origin_data_infos.copy()
                # 将gt_label等于0的项存储到一个新列表中
                selected_data = [data for data in self.data_infos if data['gt_label'] == self.CLASSES.index('negative')]
                self.data_infos = [data for data in self.data_infos if data['gt_label'] != self.CLASSES.index('negative')]
                # 从selected_data中随机选择一部分进行删除
                # num_to_delete = int(len(selected_data) * ((self.negative_count - self.positive_count) / self.negative_count))  # 删除一半的项，可以根据需求调整比例
                num_choice = int(self.positive_count * self.ratio_NP)#这里是选择保留这个数量的负样本
                items_choice = random.sample(selected_data, num_choice)
                # 从data_list中删除选定的项
                self.data_infos = items_choice + self.data_infos
                print("restructure negative")
                self.current_iterative = 1
            # print(self.current_iterative)
        return len(self.data_infos)

    def __getitem__(self, idx):

        return self.prepare_data(idx)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(expanduser(classes))
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 2)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 2))
        # print(topk)
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results
