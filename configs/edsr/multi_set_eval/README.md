# Multi interpolation evaluation

## With baseline models

|   Model   | Dataset | Interpolation  |  PSNR   |  SSIM  | Scale |
| :-------: | :-----: | :------------: | :-----: | :----: | :---: |
| [edsr_x2c64b16_1x16_300k_div2k](../edsr_x2c64b16_1xb16-300k_div2k.py) |  DIV2K  | bicubic | 30.0175 | 0.9216 | 2 |
| [edsr_x2c64b16_1x16_300k_div2k](../edsr_x2c64b16_1xb16-300k_div2k.py) |  DIV2K  | bilinear | 31.8617 | 0.9172 | 2 |
| [edsr_x2c64b16_1x16_300k_div2k](../edsr_x2c64b16_1xb16-300k_div2k.py) |  DIV2K  | nearest | 24.7872 | 0.7851 | 2 |
| [edsr_x3c64b16_1x16_300k_div2k](../edsr_x3c64b16_1xb16-300k_div2k.py) |  DIV2K  | bicubic | 26.7666 | 0.8480 | 3 |
| [edsr_x3c64b16_1x16_300k_div2k](../edsr_x3c64b16_1xb16-300k_div2k.py) |  DIV2K  | bilinear | 23.5642 | 0.7447 | 3 |
| [edsr_x3c64b16_1x16_300k_div2k](../edsr_x3c64b16_1xb16-300k_div2k.py) |  DIV2K  | nearest | 21.3182 | 0.6536 | 3 |
| [edsr_x4c64b16_1x16_300k_div2k](../edsr_x4c64b16_1xb16-300k_div2k.py) |  DIV2K  | bicubic | 25.1079 | 0.7873 | 4 |
| [edsr_x4c64b16_1x16_300k_div2k](../edsr_x4c64b16_1xb16-300k_div2k.py) |  DIV2K  | bilinear | 23.4003 | 0.7261 | 4 |
| [edsr_x4c64b16_1x16_300k_div2k](../edsr_x4c64b16_1xb16-300k_div2k.py) |  DIV2K  | nearest | 19.4297 | 0.5756 | 4 |


## With new models trained on multi interpolation resize modes augmentation dataset

|   Model   | Dataset | Interpolation  |  PSNR   |  SSIM  | Scale |
| :-------: | :-----: | :------------: | :-----: | :----: | :---: |
| [edsr_x2_div2k_multi_interp_aug](../edsr_x2c64b16_1xb16-300k_div2k_multi_sub_sampling.py) |  DIV2K  | bicubic | 33.5100 | 0.9284 | 2 |
| [edsr_x2_div2k_multi_interp_aug](../edsr_x2c64b16_1xb16-300k_div2k_multi_sub_sampling.py) |  DIV2K  | bilinear | 32.9926 | 0.9218 | 2 |
| [edsr_x2_div2k_multi_interp_aug](../edsr_x2c64b16_1xb16-300k_div2k_multi_sub_sampling.py) |  DIV2K  | nearest | 27.8316 | 0.8461 | 2 |
| [edsr_x3_div2k_multi_interp_aug](../edsr_x3c64b16_1xb16-300k_div2k_multi_sub_sampling.py) |  DIV2K  | bicubic | 30.0442 | 0.8559 | 3 |
| [edsr_x3_div2k_multi_interp_aug](../edsr_x3c64b16_1xb16-300k_div2k_multi_sub_sampling.py) |  DIV2K  | bilinear | 28.2417 | 0.8250 | 3 |
| [edsr_x3_div2k_multi_interp_aug](../edsr_x3c64b16_1xb16-300k_div2k_multi_sub_sampling.py) |  DIV2K  | nearest | 24.3707 | 0.7262 | 3 |
| [edsr_x4_div2k_multi_interp_aug](../edsr_x4c64b16_1xb16-300k_div2k_multi_sub_sampling.py) |  DIV2K  | bicubic | 28.1731 | 0.7974 | 4 |
| [edsr_x4_div2k_multi_interp_aug](../edsr_x4c64b16_1xb16-300k_div2k_multi_sub_sampling.py) |  DIV2K  | bilinear | 26.9745 | 0.7759 | 4 |
| [edsr_x4_div2k_multi_interp_aug](../edsr_x4c64b16_1xb16-300k_div2k_multi_sub_sampling.py) |  DIV2K  | nearest | 22.8511 | 0.6642 | 4 |
