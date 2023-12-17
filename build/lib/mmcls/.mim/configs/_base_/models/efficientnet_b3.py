# # model settings
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(type='EfficientNet', arch='b3'),
#     neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='LinearClsHead',
#         # num_classes=3,
#         num_classes=195,
#         in_channels=1536,
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#         topk=(1, 3),
#     ))


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
            type='EfficientNet', arch='b3',
            init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain_model/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth',
            prefix='backbone',
        )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        # num_classes=195,
        num_classes=196,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
