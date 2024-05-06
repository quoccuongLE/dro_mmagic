#!/bin/bash
mkdir -p pretrained_weights/edsr/

for SCALE in 2 3 4
do
  FILE=pretrained_weights/edsr/edsr_x${SCALE}c64b16_1xb16-300k_div2k_multi_interp_best.pth
  if [ ! -f "$FILE" ]; then
    wget https://github.com/quoccuongLE/mmagic/releases/download/model-v0.1.0/edsr_x${SCALE}c64b16_1xb16-300k_div2k_multi_interp_best.pth \
    -O pretrained_weights/edsr/edsr_x${SCALE}c64b16_1xb16-300k_div2k_multi_interp_best.pth
  fi
  for MODE in bicubic bilinear nearest
  do
    CUDA_VISIBLE_DEVICES=0 python tools/test.py \
      configs/edsr/multi_set_eval/edsr_x${SCALE}c64b16_1xb16-300k_div2k_valid_${MODE}_eval.py \
      ${FILE} --work-dir work_dirs/temp_mix_ds_eval/${MODE}_x${SCALE}
  done
done
