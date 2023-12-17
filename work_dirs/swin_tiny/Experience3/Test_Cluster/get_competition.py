# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import time
import pdb
import mmcv
import numpy as np
import torch
from tqdm import tqdm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import matplotlib.pyplot as plt

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, get_root_logger,
                         setup_multi_processes, wrap_distributed_model,
                         wrap_non_distributed_model)
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances

from sklearn.cluster import k_means
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import StandardScaler
from mmcv.parallel import collate
# from mmcls.utils import build_from_cfg
import numpy as np
from mmcls.datasets.pipelines import Compose
from openpyxl import Workbook
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def kmeans_cosine(data, k, max_iter=7):
    # 随机初始化聚类中心
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(max_iter):
        # 计算每个样本与聚类中心的相似性
        similarities = np.array([cosine_similarity(sample, centroid) for sample in data for centroid in centroids])
        similarities = similarities.reshape(len(data), k)

        # 分配样本到最近的聚类中心
        labels = np.argmax(similarities, axis=1)

        # 更新聚类中心
        new_centroids = np.array([np.mean(data[labels == i], axis=0) for i in range(k)])
        # 如果聚类中心不再改变，结束迭代
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels,centroids


def predict_cluster(data, centroids):
    # 计算新数据与聚类中心的相似性
    similarities = np.array([cosine_similarity(sample, centroid) for sample in data for centroid in centroids])
    similarities = similarities.reshape(len(data), len(centroids))

    # 排序并提取前5个最大值
    sorted_indices = np.argsort(similarities, axis=1)[:, ::-1]  # 按从大到小排序的索引
    top5_values = similarities[np.arange(len(data))[:, None], sorted_indices[:, :5]]  # 提取前5个最大值
    
    # 分配新数据到最近的聚类中心
    labels = np.argmax(similarities, axis=1)
    # max_values = np.max(similarities, axis=1)

    return labels,top5_values[0]

def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = args.device or auto_select_device()
    
    
    # cfg.data.val['data_prefix'] = "/data4/lj/Classification/1.7/work_dirs/swin_tiny/Experience3/Test_Cluster/fake"
    cfg.data.val['data_prefix'] = "/data4/lj/Dataset/JML/Experience3/Negative_folder"
    # cfg.data.val['data_prefix'] = "/data4/lj/Dataset/JML/Experience3/redundanceBox"
    cfg.data['samples_per_gpu'] = 1
    
    Cluster_dataset = build_dataset(cfg.data.val, default_args=dict(test_mode=True))

    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
        dist=False,
        round_up=True,
    )
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    cluster_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('val_dataloader', {}),
    }
    
    cluster_data_loader = build_dataloader(Cluster_dataset, **cluster_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    CLASSES = checkpoint['meta']['CLASSES']

    model = wrap_non_distributed_model(
        model, device=cfg.device, device_ids=cfg.gpu_ids)

    model.CLASSES = CLASSES
    # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
    #                             **show_kwargs)
    model.eval()
    centroids = np.loadtxt('work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_centroids_k44_new_positive.txt')
    # 创建一个新的工作簿
    workbook = Workbook()
    # 获取当前活动的工作表
    sheet = workbook.active
    insert_list = []
    for i, data in enumerate(cluster_data_loader):
        with torch.no_grad():
            test = model.module 
            print(data['img_metas'].data[0][0]['ori_filename'])
            haha = test.extract_feat(data['img'].cuda())
            label,Conf = predict_cluster(haha[0].cpu().numpy(),centroids)
            insert_list.append(data['img_metas'].data[0][0]['ori_filename'])
            insert_list.extend(Conf[i] for i in range(len(Conf)))
            insert_list.append(Conf[0]-Conf[1])
            sheet.append(insert_list)
            insert_list = []
            # print(label,Conf)
    # # 保存工作簿到文件
    workbook.save('Negative_folder.xlsx')
if __name__ == '__main__':
    main()