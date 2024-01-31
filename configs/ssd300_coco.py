# _base_ = [
#     '/workspace/odt/configs/_base_/models/ssd300.py', '/workspace/odt/configs/_base_/datasets/coco_detection.py',
#     '/workspace/odt/configs/_base_/schedules/schedule_2x.py', '/workspace/odt/configs/_base_/default_runtime_ssd300.py'
# ]

# data_root = '/workspace/odt/SSD/datasets/bdd100k/'

# dataset_type = 'CocoDataset'

# # List of object categories / compatible for BDD 100K
# classes = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign')
#
#
# img_norm_cfg=dict(
#         mean=[123.675, 116.28, 103.53],
#         std=[1, 1, 1],
#         to_rgb=True)

# # dataset settings
# input_size = 300
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='Expand',
#         mean=[123.675, 116.28, 103.53],
#         to_rgb=True,
#         ratio_range=(1, 4)),
#     dict(
#         type='MinIoURandomCrop',
#         min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
#         min_crop_size=0.3),
#     dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=1),
    
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# # test_pipeline = [
# #     dict(type='LoadImageFromFile'),
# #     # dict(type='LoadAnnotations', with_bbox=True),
# #     dict(
# #         type='MultiScaleFlipAug',
# #         img_scale = (input_size, input_size),
# #         flip=False,
# #         transforms=[
# #             dict(type='Resize', keep_ratio=False),
# #             dict(type='RandomFlip'),
# #             dict(type='Normalize', **img_norm_cfg),
# #             dict(type='Pad', size_divisor=1),
# #             dict(type='ImageToTensor', keys=['img']),
# #             dict(type='Collect', keys=['img']),
# #         ])    
# # ]

# # Configure train dataset
# train_dataset = dict(
#     type='MultiImageMixDataset',    # data augmentation with mosaic and mixup
#     dataset=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file=data_root + 'coco_labels/det_train_coco.json', # path of ground truth info 
#         img_prefix=data_root + 'images/100k/train/',    # path of images
        
#         # define preprocessing script list
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True)    # load ground truth object location & class info.
#         ],
#         filter_empty_gt=False, # No object in image, exlcude image from dataset
#     ),
#     pipeline=train_pipeline)

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',   # data augmentation with scale, flip
#         img_scale=(input_size, input_size),
#         flip=False,
#         transforms=[    # additional preprocessing sequences
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(
#                 type='Pad',
#                 pad_to_square=True, # square padding
#                 pad_val=dict(img=(114.0, 114.0, 114.0))),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img'])
#         ])
# ]

# # Configure validation / test dataset
# data = dict(
#     samples_per_gpu=3,
#     workers_per_gpu=4,
#     persistent_workers=True,    # run worker in background

#     train=train_dataset,
#     val=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file=data_root + 'coco_labels/det_val_coco.json',
#         img_prefix=data_root + 'images/100k/val/',
#         pipeline=test_pipeline),

#     test=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file=data_root + 'coco_labels/det_val_coco.json',
#         img_prefix=data_root + 'images/100k/val/',
#         pipeline=test_pipeline))


# # optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4))

# # # Add to avoid NaN output by KH
# # optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# custom_hooks = [
#     dict(type='NumClassCheckHook'),
#     # dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
# ]

# # NOTE: `auto_scale_lr` is for automatically scaling LR,
# # USER SHOULD NOT CHANGE ITS VALUES.
# # base_batch_size = (8 GPUs) x (8 samples per GPU)
# auto_scale_lr = dict(base_batch_size=64)

""""""""""""""

_base_ = [
    '/workspace/odt/configs/_base_/models/ssd300.py', '/workspace/odt/configs/_base_/datasets/coco_detection.py',
    '/workspace/odt/configs/_base_/schedules/schedule_2x.py', '/workspace/odt/configs/_base_/default_runtime_ssd300.py'
]

data_root = '/workspace/odt/SSD/datasets/bdd100k/'
dataset_type = 'CocoDataset'


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'coco_labels/det_train_coco.json',
            img_prefix=data_root + 'images/100k/train/',
            pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
# optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
# optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
    
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)