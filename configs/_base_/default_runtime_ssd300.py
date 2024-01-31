# # default_scope = 'mmdet'

# # Control the frequency of saving model checkpoints during training
# checkpoint_config = dict(interval=1)

# # Configures the evaluation process
# evaluation = dict(save_best='auto', dynamic_intervals=[(300, 1)], interval=1, metric=['bbox'])


# # distribution learing
# opencv_num_threads = 0
# mp_start_method = 'fork'
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = ''  # load pretrained file
# resume_from = None
# workflow = [('train', 1)]   # perform training 1 time

# # Sets the frequency of logging training info.    
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook')    # save train data as .txt
#     ])
# custom_hooks = [dict(type='NumClassCheckHook')] # check no.class at start of training

""""""""""""""
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/workspace/odt/work_dirs/ssd300_coco/latest.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)