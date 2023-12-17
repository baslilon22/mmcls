data_root = '/data3/hjz/datasets/aiot/'
work_dir = '/data/data2/linjie/train/work_dirs/test'
num_classes = 788
img_size = 128
samples_per_gpu = 170
checkpoint = '/data/data2/linjie/train/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='tiny',
        img_size=128,
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/data/data2/linjie/train/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth',
            prefix='backbone'),
        pad_small_map=False,
        stage_cfgs=dict(block_cfgs=dict(window_size=4))),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=788,
        in_channels=768,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys=dict({
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    }))
optimizer = dict(
    type='AdamW',
    lr=9.960937500000001e-05,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })))
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_ratio=0.001,
    warmup_iters=20,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/data/data2/linjie/train/best_accuracy_top-1_epoch_54.pth'
resume_from = None
workflow = [('train', 1)]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='LoadImageFromLmdb',
        file_client_args=dict(
            lmdb_list='/data/data2/linjie/train/Temp/train_aiot.lmdb')),
    dict(
        type='ScaleCenterCrop',
        p=0.5,
        crop_range=(0.16666666666666663, 0.33333333333333326)),
    dict(type='PaddingShortBorder'),
    dict(
        type='Resize',
        size=128,
        backend='pillow',
        interpolation='bicubic',
        adaptive_side='long'),
    dict(type='Rotate', angle=8.0, interpolation='bicubic'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='LoadImageFromLmdb',
        file_client_args=dict(
            lmdb_list='/data/data2/linjie/train/Temp/test_aiot.lmdb')),
    dict(type='PaddingShortBorder'),
    dict(
        type='Resize',
        size=128,
        backend='pillow',
        interpolation='bicubic',
        adaptive_side='long'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
dataset_type = 'CustomDataset'
data = dict(
    samples_per_gpu=170,
    workers_per_gpu=1,
    train=dict(
        type='CustomDataset',
        classes='/data/data2/linjie/train/Temp/train/labels.txt',
        data_prefix='/data3/hjz/datasets/aiot/',
        ann_file='/data/data2/linjie/train/Temp/train/train.txt',
        pipeline=[
            dict(
                type='LoadImageFromLmdb',
                file_client_args=dict(
                    lmdb_list='/data/data2/linjie/train/Temp/train_aiot.lmdb')
            ),
            dict(
                type='ScaleCenterCrop',
                p=0.5,
                crop_range=(0.16666666666666663, 0.33333333333333326)),
            dict(type='PaddingShortBorder'),
            dict(
                type='Resize',
                size=128,
                backend='pillow',
                interpolation='bicubic',
                adaptive_side='long'),
            dict(type='Rotate', angle=8.0, interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CustomDataset',
        classes='/data/data2/linjie/train/Temp/train/labels.txt',
        data_prefix='/data3/hjz/datasets/aiot/',
        ann_file='/data/data2/linjie/train/Temp/train/test.txt',
        pipeline=[
            dict(
                type='LoadImageFromLmdb',
                file_client_args=dict(
                    lmdb_list='/data/data2/linjie/train/Temp/test_aiot.lmdb')),
            dict(type='PaddingShortBorder'),
            dict(
                type='Resize',
                size=128,
                backend='pillow',
                interpolation='bicubic',
                adaptive_side='long'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='CustomDataset',
        classes='/data/data2/linjie/train/Temp/train/labels.txt',
        data_prefix='/data3/hjz/datasets/aiot/',
        ann_file='/data/data2/linjie/train/Temp/train/test.txt',
        pipeline=[
            dict(
                type='LoadImageFromLmdb',
                file_client_args=dict(
                    lmdb_list='/data/data2/linjie/train/Temp/test_aiot.lmdb')),
            dict(type='PaddingShortBorder'),
            dict(
                type='Resize',
                size=128,
                backend='pillow',
                interpolation='bicubic',
                adaptive_side='long'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
gpu_ids = range(0, 6)
evaluation = dict(interval=1, metric='accuracy', save_best='auto')
