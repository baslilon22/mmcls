# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv import FileClient

from .base_dataset import BaseDataset
from .builder import DATASETS
import random

from tqdm import tqdm
def find_folders(root: str,
                 file_client: FileClient) -> Tuple[List[str], Dict[str, int]]:
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        Tuple[List[str], Dict[str, int]]:

        - folders: The name of sub folders under the root.
        - folder_to_idx: The map from folder name to class idx.
    """
    folders = list(
        file_client.list_dir_or_file(
            root,
            list_dir=True,
            list_file=False,
            recursive=False,
        ))
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folders, folder_to_idx


def get_samples(root: str, folder_to_idx: Dict[str, int],
                is_valid_file: Callable, file_client: FileClient):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        is_valid_file (Callable): A function that takes path of a file
            and check if the file is a valid sample file.

    Returns:
        Tuple[list, set]:

        - samples: a list of tuple where each element is (image, class_idx)
        - empty_folders: The folders don't have any valid files.
    """
    samples = []
    available_classes = set()

    # # for folder_name in sorted(list(folder_to_idx.keys())):
    # for folder_name in tqdm(sorted(list(folder_to_idx.keys()))):
    #     # if folder_name != "negative":
    #     if folder_name.split("_")[:-1] == ['xw','label']:
    #         _dir = file_client.join_path(root, folder_name)
    #         files = list(
    #             file_client.list_dir_or_file(
    #                 _dir,
    #                 list_dir=False,
    #                 list_file=True,
    #                 recursive=True,
    #             ))
    #         for file in sorted(list(files)):
    #             if is_valid_file(file):
    #                 path = file_client.join_path(folder_name, file)
    #                 item = (path, folder_to_idx[folder_name])
    #                 samples.append(item)
    #                 available_classes.add(folder_name)
    
    #origine            
    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = file_client.join_path(root, folder_name)
        files = list(
            file_client.list_dir_or_file(
                _dir,
                list_dir=False,
                list_file=True,
                recursive=True,
            ))
        for file in sorted(list(files)):
            if is_valid_file(file):
                path = file_client.join_path(folder_name, file)
                item = (path, folder_to_idx[folder_name])
                samples.append(item)
                available_classes.add(folder_name)
                    
    # folder_name = "negative"
    # _dir = file_client.join_path(root, folder_name)
    # files = list(
    #     file_client.list_dir_or_file(
    #         _dir,
    #         list_dir=False,
    #         list_file=True,
    #         recursive=True,
    #     ))
    # #它是不是每一次循环都会执行这个？好像不是？？？？？
    # files = random.sample(files, int(Positive_count / 2))
    # # print("sample")
    # for file in sorted(list(files)):
    #     if is_valid_file(file):
    #         path = file_client.join_path(folder_name, file)
    #         item = (path, folder_to_idx[folder_name])
    #         #在这里进行一些处理。
    #         samples.append(item)
    #         available_classes.add(folder_name)
    # print("test")
                    
    # for folder_name in sorted(list(folder_to_idx.keys())):
    #     _dir = file_client.join_path(root, folder_name)
    #     files = list(
    #         file_client.list_dir_or_file(
    #             _dir,
    #             list_dir=False,
    #             list_file=True,
    #             recursive=True,
    #         ))
    #     for file in sorted(list(files)):
    #         if is_valid_file(file):
    #             path = file_client.join_path(folder_name, file)
    #             item = (path, folder_to_idx[folder_name])
    #             samples.append(item)
    #             available_classes.add(folder_name)

    empty_folders = set(folder_to_idx.keys()) - available_classes

    return samples, empty_folders


@DATASETS.register_module()
class CustomDataset(BaseDataset):
    """Custom dataset for classification.

    The dataset supports two kinds of annotation format.

    1. An annotation file is provided, and each line indicates a sample:

       The sample files: ::

           data_prefix/
           ├── folder_1
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           └── folder_2
               ├── 123.png
               ├── nsdf3.png
               └── ...

       The annotation file (the first column is the image path and the second
       column is the index of category): ::

            folder_1/xxx.png 0
            folder_1/xxy.png 1
            folder_2/123.png 5
            folder_2/nsdf3.png 3
            ...

       Please specify the name of categories by the argument ``classes``.

    2. The samples are arranged in the specific way: ::

           data_prefix/
           ├── class_x
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           │       └── xxz.png
           └── class_y
               ├── 123.png
               ├── nsdf3.png
               ├── ...
               └── asd932_.png

    If the ``ann_file`` is specified, the dataset will be generated by the
    first way, otherwise, try the second way.

    Args:
        data_prefix (str): The path of data directory.
        pipeline (Sequence[dict]): A list of dict, where each element
            represents a operation defined in :mod:`mmcls.datasets.pipelines`.
            Defaults to an empty tuple.
        classes (str | Sequence[str], optional): Specify names of classes.

            - If is string, it should be a file path, and the every line of
              the file is a name of a class.
            - If is a sequence of string, every item is a name of class.
            - If is None, use ``cls.CLASSES`` or the names of sub folders
              (If use the second way to arrange samples).

            Defaults to None.
        ann_file (str, optional): The annotation file. If is string, read
            samples paths from the ann_file. If is None, find samples in
            ``data_prefix``. Defaults to None.
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        test_mode (bool): In train mode or test mode. It's only a mark and
            won't be used in this class. Defaults to False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            If None, automatically inference from the specified path.
            Defaults to None.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 Batch_size=0, #后来加的
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.file_client_args = file_client_args

        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            Batch_size=Batch_size,
            classes=classes,
            ann_file=ann_file,
            test_mode=test_mode)

    def _find_samples(self):
        """find samples from ``data_prefix``."""
        file_client = FileClient.infer_client(self.file_client_args,
                                              self.data_prefix)
        classes, folder_to_idx = find_folders(self.data_prefix, file_client)
        samples, empty_classes = get_samples(
            self.data_prefix,
            folder_to_idx,
            is_valid_file=self.is_valid_file,
            file_client=file_client,
        )

        if len(samples) == 0:
            raise RuntimeError(
                f'Found 0 files in subfolders of: {self.data_prefix}. '
                f'Supported extensions are: {",".join(self.extensions)}')

        if self.CLASSES is not None:
            assert len(self.CLASSES) == len(classes), \
                f"The number of subfolders ({len(classes)}) doesn't match " \
                f'the number of specified classes ({len(self.CLASSES)}). ' \
                'Please check the data folder.'
        else:
            self.CLASSES = classes

        if empty_classes:
            warnings.warn(
                'Found no valid file in the folder '
                f'{", ".join(empty_classes)}. '
                f"Supported extensions are: {', '.join(self.extensions)}",
                UserWarning)

        self.folder_to_idx = folder_to_idx

        return samples

    def load_annotations(self):
        """Load image paths and gt_labels."""
        if self.ann_file is None:
            samples = self._find_samples()
        elif isinstance(self.ann_file, str):
            lines = mmcv.list_from_file(
                self.ann_file, file_client_args=self.file_client_args)
            samples = [x.strip().rsplit(' ', 1) for x in lines]
        else:
            raise TypeError('ann_file must be a str or None')

        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)
