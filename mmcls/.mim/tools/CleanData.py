# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number
import pdb
import mmcv
import numpy as np
from tqdm import tqdm
import torch
from mmcv import DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import os.path as osp
from mmcv.image import tensor2imgs
# from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, get_root_logger,
                         setup_multi_processes, wrap_distributed_model,
                         wrap_non_distributed_model)
import shutil

import cv2
def Pad(image):
    pad_size_h = 500
    pad_size_w = 500
    img = image
    h, w = img.shape[:2]
    
    if h > pad_size_h or w > pad_size_w:
        if h > w:
            scale = pad_size_h / h
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = cv2.resize(img,(new_w,new_h))
        else:
            scale = pad_size_w / w
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = cv2.resize(img,(new_w,new_h))
    else:
        new_h = h
        new_w = w    
    # 计算要添加的行和列数量
    pad_h = max(0, pad_size_h - new_h)
    pad_w = max(0, pad_size_w - new_w)
    h_start = pad_h // 2
    w_start = pad_w // 2

    # 创建填充后的图像
    padded_image = np.zeros((pad_size_h, pad_size_w, 3), dtype=img.dtype)
    padded_image[h_start:h_start+new_h, w_start:w_start+new_w, :] = img
    
    # padded_image = np.zeros((pad_size_h,pad_size_w,3),dtype=img.dtype)
    
    # h_start = (pad_size_h - h) // 2
    # w_start = (pad_size_w - w) // 2
    # padded_image[h_start:h_start+h,w_start:w_start+w,:] = img
    
    results= padded_image
    # results['img_shape'] = padded_image.shape
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', help='device used for testing')
    args = parser.parse_args()


    return args

def single_gpu_test(model,
                    data_loader):
    # 打开文件以写入模式
    file = open("example.txt", "w")
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            for j,item in enumerate(result):
                label_class = data['img_metas'].data[0][j]['ori_filename'].split('/')[0]
                if label_class == "qualified":
                    score = result[j][0]
                elif label_class == "unqualified":
                    score = result[j][1]
                else:
                    print("error")    
                
                file.write(data['img_metas'].data[0][j]['filename']+" "+str(score)+"\n")  # 使用换行符\n进行换行           
    # 关闭文件
    file.close()
    return results


def get_txt():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    cfg.device = args.device or auto_select_device()

    # init distributed env first, since logger depends on the dist info.

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
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
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

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


    model = wrap_non_distributed_model(
        model, device=cfg.device, device_ids=cfg.gpu_ids)
    if cfg.device == 'ipu':
        from mmcv.device.ipu import cfg2options, ipu_model_wrapper
        opts = cfg2options(cfg.runner.get('options_cfg', {}))
        if fp16_cfg is not None:
            model.half()
        model = ipu_model_wrapper(model, opts, fp16_cfg=fp16_cfg)
        data_loader.init(opts['inference'])
    model.CLASSES = CLASSES
    outputs = single_gpu_test(model, data_loader)
    

def sort_picture():
    with open('unqualified.txt', 'r') as file:
        destination_path = "/data4/lj/forgery_Detect/newly_sort_unqualified"
        lines = file.readlines()
        for i, line in enumerate(tqdm(lines, desc='Processing lines')):
            
            line = line.strip()  # 去除行末尾的换行符和空白字符
            split = line.split(" ")
            file_path = split[0]
            file_extension = os.path.splitext(file_path)[1]
            score = split[1][:6]
            destination_file_path = os.path.join(destination_path,score+"_"+str(i)+file_extension)
            shutil.copy2(file_path, destination_file_path)
            
            # 在这里对每行内容进行处理或打印
            # print(line)
    with open('qualified.txt', 'r') as file:
        destination_path = "/data4/lj/forgery_Detect/newly_sort_qualified"
        lines = file.readlines()
        for i, line in enumerate(tqdm(lines, desc='Processing lines')):
            
            line = line.strip()  # 去除行末尾的换行符和空白字符
            split = line.split(" ")
            file_path = split[0]
            file_extension = os.path.splitext(file_path)[1]
            score = split[1][:6]
            destination_file_path = os.path.join(destination_path,score+"_"+str(i)+file_extension)
            shutil.copy2(file_path, destination_file_path)
            
            # 在这里对每行内容进行处理或打印
            # print(line)

if __name__ == '__main__':
    # get_txt()
    sort_picture()
