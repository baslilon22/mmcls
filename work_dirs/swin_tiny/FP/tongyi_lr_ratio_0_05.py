train_data_root = '/data4/lj/forgery_Detect/Forgery_train_20231106'
test_data_root = '/data4/lj/forgery_Detect/Forgery_test_20231106'
work_dir = 'work_dirs/swin_tiny/FP'
num_classes = 2
batch_size = 70
max_epochs = 200
interval_save = 50
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='tiny',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'pretrain_model/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth',
            prefix='backbone'),
        img_size=224,
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=768,
        init_cfg=None,
        loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25, reduction='mean'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])
dataset_type = 'CustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Rotate', angle=8.0, interpolation='bicubic'),
    dict(
        type='Rotate',
        angle=90.0,
        interpolation='bicubic',
        prob=0.1,
        random_negative_prob=0.5),
    dict(
        type='Rotate',
        angle=180.0,
        interpolation='bicubic',
        prob=0.1,
        random_negative_prob=0.5),
    dict(type='Resize', size=224),
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
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=190,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_prefix='/data4/lj/forgery_Detect/Forgery_train_20231106',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Rotate', angle=8.0, interpolation='bicubic'),
            dict(
                type='Rotate',
                angle=90.0,
                interpolation='bicubic',
                prob=0.1,
                random_negative_prob=0.5),
            dict(
                type='Rotate',
                angle=180.0,
                interpolation='bicubic',
                prob=0.1,
                random_negative_prob=0.5),
            dict(type='Resize', size=224),
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
        data_prefix='/data4/lj/forgery_Detect/Forgery_test_20231106',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=224),
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
        data_prefix='/data4/lj/forgery_Detect/Forgery_test_20231106',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=10, metric='accuracy')
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys=dict({
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    }))
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.01,
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
    min_lr_ratio=0.005,
    warmup='linear',
    warmup_ratio=0.001,
    warmup_iters=20,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=10, save_optimizer=True)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = [0]
