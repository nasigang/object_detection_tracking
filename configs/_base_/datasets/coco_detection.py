# # # dataset settings
# dataset_type = 'CocoDataset'
# data_root = '/workspace/odt/SSD/datasets/bdd100k/'

# backend_args = None

# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]
	
# data = dict(
#     val=dict(
#         type='CocoDataset',
#         ann_file=data_root + 'annotations/instances_val2017.json'))

""""""""""""""
import mmdet

# dataset settings
data_root = '/workspace/odt/SSD/datasets/bdd100k/'
dataset_type = 'CocoDataset'

# List of object categories / compatible for BDD 100K
classes = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign')
mmdet.datasets.coco.CocoDataset.CLASSES=classes

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
            ann_file=data_root + 'coco_labels/det_train_coco.json',
            img_prefix=data_root + 'images/100k/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
            ann_file=data_root + 'coco_labels/det_val_coco.json',
            img_prefix=data_root + 'images/100k/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
            ann_file=data_root + 'coco_labels/det_val_coco.json',
            img_prefix=data_root + 'images/100k/val/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')