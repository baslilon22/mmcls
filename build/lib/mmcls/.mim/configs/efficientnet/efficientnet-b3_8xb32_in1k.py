# _base_ = [
#     '../_base_/models/efficientnet_b3.py',
#     '../_base_/datasets/imagenet_bs32.py',
#     '../_base_/schedules/imagenet_bs256.py',
#     '../_base_/default_runtime.py',
# ]
# # from mmengine.registry import TRANSFORMS

# model = dict(
#     backbone=dict(
#         frozen_stages=2,
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='/home/common/linjie/Classification/1.1/pretrain_model/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth',
#             prefix='backbone',
#         )),
#     # head=dict(num_classes=3),
#     head=dict(num_classes=195),
# )
# checkpoint=dict(type='CheckpointHook', interval=-1, save_best='auto'),
# # evaluation = dict(interval=1,metric='accuracy',save_best='accuracy/top1')
# # dataset settings
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     # dict(type='Pad', pad_size_h = 600,pad_size_w = 600),
#     # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='PackInputs'),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     # dict(type='EfficientNetCenterCrop', crop_size=600),
#     dict(type='PackInputs'),
# ]

# train_dataloader = dict(dataset=dict(
#     type='CustomDataset',
#     # data_prefix='/home/common/linjie/Cls_Dataset/train',
#     data_prefix='/home/common/linjie/Cls_Camera_sk_sku/train',
#     # data_prefix='/home/common/linjie/Mask_Cls_Dataset/Mask_10/train',
# 	with_label=True,
#     pipeline=train_pipeline
#     )
# )
# val_dataloader = dict(dataset=dict(
#     type='CustomDataset',
#     # data_prefix='/home/common/linjie/Cls_Dataset/val',
#     data_prefix='/home/common/linjie/Cls_Camera_sk_sku/val',
#     # data_prefix='/home/common/linjie/Mask_Cls_Dataset/Mask_10/val',
# 	with_label=True,
#     pipeline=test_pipeline
#     )
# )
# test_dataloader = val_dataloader


_base_ = [
    '../_base_/models/efficientnet_b3.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]
# model = dict(
#     backbone=dict(
#         frozen_stages=2,
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='pretrain_model/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth',
#             prefix='backbone',
#         )),
#     # head=dict(num_classes=3),
#     head=dict(num_classes=195),
# )
# dataset settings
# dataset_type = 'ImageNet'
dataset_type = 'CustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(
    #     type='RandomResizedCrop',
    #     size=300,
    #     efficientnet_style=True,
    #     interpolation='bicubic'),
    dict(type='Jie_Pad', pad_size_h= 500, pad_size_w=500),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),#暂时不能注释，注释的话就会报类型错误
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(
    #     type='CenterCrop',
    #     crop_size=300,
    #     efficientnet_style=True,
    #     interpolation='bicubic'),
    # dict(type='Pad', pad_size_h = 600,pad_size_w = 600),
    dict(type='Jie_Pad', pad_size_h= 500, pad_size_w=500),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))





checkpoint=dict(type='CheckpointHook', interval=-1, save_best='auto'),


