import os

_base_ = ["../edsr_x3c64b16_1xb16-300k_div2k_multi_sub_sampling.py"]


scale = 3
dataset_type = "BasicImageDataset"
data_root = os.environ.get("DSDIR", "/media/Data2-HDD8/datasets")

interp_mode = "bilinear"
experiment_name = f"edsr_x{scale}c64b16_1xb16-300k_div2k_valid_{interp_mode}_multi_interp"
work_dir = f"./work_dirs/{experiment_name}"

test_pipeline = [
    dict(type="LoadImageFromFile", key="img", color_type="color", channel_order="rgb", imdecode_backend="cv2"),
    dict(type="LoadImageFromFile", key="gt", color_type="color", channel_order="rgb", imdecode_backend="cv2"),
    dict(type="PackInputs"),
]

test_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type="set5", task_name="sisr", group_id=0),
        data_root=data_root + "/DIV2K",
        data_prefix=dict(img=f"DIV2K_valid_LR_{interp_mode}/X{scale}", gt="DIV2K_valid_HR"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = [test_dataloader]

val_evaluator = dict(
    type="Evaluator",
    metrics=[
        dict(type="MAE"),
        dict(type="PSNR", crop_border=scale),
        dict(type="SSIM", crop_border=scale),
    ],
)

test_evaluator = [val_evaluator]
