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


import numpy as np

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def kmeans_cosine(data, k, max_iter=3):
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
        print(centroids-new_centroids)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels,centroids


def predict_cluster(data, centroids):
    # 计算新数据与聚类中心的相似性
    similarities = np.array([cosine_similarity(sample, centroid) for sample in data for centroid in centroids])
    similarities = similarities.reshape(len(data), len(centroids))

    # 分配新数据到最近的聚类中心
    labels = np.argmax(similarities, axis=1)

    return labels


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

    # 先注释
    
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


    # dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))
    
    # Cluster_dataset = build_dataset(cfg.data.val, default_args=dict(test_mode=True))
    Cluster_dataset = build_dataset(cfg.data.val, default_args=dict(test_mode=False))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
        dist=False,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    cluster_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('val_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    # data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    cluster_data_loader = build_dataloader(Cluster_dataset, **cluster_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES
        
    # print(CLASSES)


    model = wrap_non_distributed_model(
        model, device=cfg.device, device_ids=cfg.gpu_ids)

    model.CLASSES = CLASSES
    # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
    #                             **show_kwargs)
    model.eval()
    results = []
    dataset = cluster_data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    sum_features = np.empty((0,768)) #70是batch_size
    
    # # 创建一个txt文档
    # # file_path = 'work_dirs/swin_tiny/Experience3/Test_Cluster/negative_example.txt'
    # # file = open(file_path, 'w')

    # #sum_features = np.empty((0,3,224,224)) #70是batch_size
    start_time = time.time()
    for i, data in enumerate(cluster_data_loader):
        # 写入内容
        # for j, one_picture in enumerate(data['img_metas'].data[0]):
        #     file.write(one_picture['ori_filename']+'\n')

        with torch.no_grad():
            test = model.module
            #这里保留一个疑问,这个data['img'].cuda()不需要经过处理嘛,直接可以进行计算:可以的,这个最大值和最小值与推理的一致
            haha = test.extract_feat(data['img'].cuda())
            # .extract_feat(data['img'])
            sum_features = np.append(sum_features,haha[0].cpu().numpy(),axis=0)
        
        prog_bar.update(data['img'].size(0))
    # # # 关闭文件
    # # file.close()
    

    np.save("work_dirs/swin_tiny/Experience3/Test_Cluster/new_positive_sum_features.npy", sum_features)

    # extract_feature = np.load("work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_extract_feature.npy")
    # labels,centroids = kmeans_cosine(extract_feature, 4000)
    # np.savetxt('work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_positive_labels_k4000.txt', labels, fmt='%d')
    # np.savetxt('work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_positive_centroids_k4000.txt', centroids)

    # extract_feature = np.load("work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_extract_feature.npy")
    # labels,centroids = kmeans_cosine(extract_feature, 2000)
    # np.savetxt('work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_positive_labels_k2000.txt', labels, fmt='%d')
    # np.savetxt('work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_positive_centroids_k2000.txt', centroids)





    # extract_feature = np.load("work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_extract_feature.npy")
    # labels,centroids = kmeans_cosine(extract_feature, 1000)
    # # 保存聚类信息到文件
    # np.savetxt('work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_sum_labels_k1000.txt', labels, fmt='%d')
    # np.savetxt('work_dirs/swin_tiny/Experience3/Test_Cluster/Super_model_centroids_k1000.txt', centroids)
    
    
    # extract_feature = np.load("work_dirs/swin_tiny/Experience3/Test_Cluster/my_model_extract_freature.npy")
    # labels,centroids = kmeans_cosine(extract_feature, 100)
    # # 保存聚类信息到文件
    # np.savetxt('work_dirs/swin_tiny/Experience3/Test_Cluster/my_model_sum_labels_k100.txt', labels, fmt='%d')
    # np.savetxt('work_dirs/swin_tiny/Experience3/Test_Cluster/my_model_centroids_k100.txt', centroids)

    # # 读取保存的聚类信息
    # saved_labels = np.loadtxt('work_dirs/swin_tiny/Experience3/Test_Cluster/labels.txt', dtype=np.int)
    # saved_centroids = np.loadtxt('work_dirs/swin_tiny/Experience3/Test_Cluster/centroids.txt')

    # start_time = time.time()
    # for i, data in enumerate(cluster_data_loader):
    #     with torch.no_grad():
    #         test = model.module
    #         haha = test.extract_feat(data['img'].cuda())
    #         label1 = predict_cluster(haha[0].cpu().numpy(),centroids)

    # end_time = time.time()
    # print("2")
    # print(end_time - start_time)            
    # print(label1)
    
    
    # K = 46  # 聚类簇的数量
    # kmeans = KMeans(n_clusters=K)
    # kmeans.fit(sum_features)
    # # K = 46  # 聚类簇的数量
    # kmeans = KMeans(n_clusters=K)
    # kmeans.fit(sum_features)

    # # 预测每个特征向量所属的簇
    # labels = kmeans.predict(sum_features)
    # # 使用t-SNE进行降维
    # tsne = TSNE(n_components=2)
    # features_tsne = tsne.fit_transform(sum_features)

    # # 绘制散点图
    # plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels)
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.title('K-means Clustering with t-SNE Visualization')
    # plt.savefig('scatter_plot_cos.png')        

    # # 将sum_features作为输入进行K-means聚类
    # k = 46  # 聚类的数量
    # kmeans = KMeans(n_clusters=k)
    # clusters = kmeans.fit_predict(sum_features)
        
    # # 对于一个新的[1, 768]特征进行预测
    # new_feature = np.random.rand(1, 768)  # 假设新的特征是随机生成的
    # nearest_cluster_idx = np.argmin(distances)

    # print("预测的最近聚类簇索引:", nearest_cluster_idx)
        
    # haha_normalized = sum_features / np.linalg.norm(sum_features, axis=1, keepdims=True)
    # similarity_matrix = cosine_similarity(haha_normalized)
    # kmeans = KMeans(n_clusters=46)
    # kmeans.fit(similarity_matrix)
    # # kmeans = KMeans(n_clusters=46)
    # # kmeans.fit(sum_features)

    # 创建KMeans对象并进行聚类

    # # 预测每个特征向量所属的簇
    # labels = kmeans.labels_

    # # 使用t-SNE进行降维
    # tsne = TSNE(n_components=2)
    # features_tsne = tsne.fit_transform(sum_features)

    # # 绘制散点图
    # plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels)
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.title('K-means Clustering with t-SNE Visualization')
    # # 保存图像到文件
    # plt.savefig('scatter_plot_cos.png')
if __name__ == '__main__':
    main()
