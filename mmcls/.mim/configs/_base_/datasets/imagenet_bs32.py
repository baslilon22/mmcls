
# dataset settings
dataset_type = 'CustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = []
test_pipeline = []
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    # persistent_workers=False,
    train=dict(
        type=dataset_type,
        data_prefix='/home/common/linjie/Cls_Camera_sk_sku/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/common/linjie/Cls_Camera_sk_sku/val',
        # ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/home/common/linjie/Cls_Camera_sk_sku/val',
        # ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')


# train_dataloader = dict(
#     batch_size=16,
#     dataset=dict(
#         type='CustomDataset',
#         data_root="/home/common/linjie/Cls_Dataset/train",
#         # ann_file='',
#         data_prefix='train',
#     ))
# val_dataloader = dict(
#     batch_size=16,
#     dataset=dict(
#         type='CustomDataset',
#         data_root="/home/common/linjie/Cls_Dataset/val",
#         # ann_file='',
#         data_prefix='test',
#     ))
# test_dataloader = val_dataloader