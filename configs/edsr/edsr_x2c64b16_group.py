import os

_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/sisr_x2_test_config.py'
]

experiment_name = 'edsr_x2c64b16_1xb16-300k_div2k_multi_subsampling_eval'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 2
# model settings
model = dict(
    type="BaseEditModelGroup",
    generator=dict(
        type="EDSRNet",
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale,
        res_scale=1,
        rgb_mean=[0.4488, 0.4371, 0.4040],
        rgb_std=[1.0, 1.0, 1.0],
    ),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type="DataPreprocessor",
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    ),
)

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=96),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackInputs')
]
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackInputs')
]

# dataset settings
dataset_type = 'GroupImageDataset'
data_root = os.environ.get("DSDIR", "/media/Data2-HDD8/datasets")

train_dataloader = dict(
    num_workers=8,
    batch_size=16,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type=dataset_type,
                ann_file="meta_info_DIV2K800sub_GT.txt",
                metainfo=dict(dataset_type="div2k", task_name="sisr", group_id=0),
                data_root=data_root + "/DIV2K",
                data_prefix=dict(img=f"DIV2K_train_LR_bicubic/X{scale}_sub", gt="DIV2K_train_HR_sub"),
                filename_tmpl=dict(img="{}", gt="{}"),
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                ann_file="meta_info_DIV2K800sub_GT.txt",
                metainfo=dict(dataset_type="div2k", task_name="sisr", group_id=0),
                data_root=data_root + "/DIV2K",
                data_prefix=dict(img=f"DIV2K_train_LR_bilinear/X{scale}_sub", gt="DIV2K_train_HR_sub"),
                filename_tmpl=dict(img="{}", gt="{}"),
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                ann_file="meta_info_DIV2K800sub_GT.txt",
                metainfo=dict(dataset_type="div2k", task_name="sisr", group_id=0),
                data_root=data_root + "/DIV2K",
                data_prefix=dict(img=f"DIV2K_train_LR_nearest/X{scale}_sub", gt="DIV2K_train_HR_sub"),
                filename_tmpl=dict(img="{}", gt="{}"),
                pipeline=train_pipeline,
            ),
        ],
    ),
)

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type="set5", task_name="sisr", group_id=0),
        # group_id=0,
        data_root=data_root + "/Set5",
        data_prefix=dict(img=f"LRbicx{scale}", gt="GTmod12"),
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='MAE'),
        dict(type='PSNR', crop_border=scale),
        dict(type='SSIM', crop_border=scale),
    ])

test_dataloader = [val_dataloader]
test_evaluator = [val_evaluator]


train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=5000)
val_cfg = dict(type='MultiValLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
