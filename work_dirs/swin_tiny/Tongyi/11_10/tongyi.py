train_data_root = '/data4/lj/Dataset/TongYi/Cls_Standard_11_10'
test_data_root = '/data4/lj/Dataset/TongYi/cls_test'
work_dir = 'work_dirs/swin_tiny/Tongyi/11_10'
num_classes = 149
batch_size = 180
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
            '/data4/lj/Classification/1.7/work_dirs/swin_tiny/Experience3/Test_Cluster/best_accuracy_top-1_epoch_36.pth',
            prefix='backbone'),
        img_size=224,
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=149,
        in_channels=768,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
            loss_weight=[
                0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1
            ]),
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
    dict(type='Jie_Pad', pad_size_h=224, pad_size_w=224),
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
    dict(type='Jie_Pad', pad_size_h=224, pad_size_w=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=180,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_prefix='/data4/lj/Dataset/TongYi/Cls_Standard_11_10',
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
            dict(type='Jie_Pad', pad_size_h=224, pad_size_w=224),
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
        data_prefix='/data4/lj/Dataset/TongYi/cls_test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Jie_Pad', pad_size_h=224, pad_size_w=224),
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
        data_prefix='/data4/lj/Dataset/TongYi/cls_test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Jie_Pad', pad_size_h=224, pad_size_w=224),
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
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=50, save_optimizer=True)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)
