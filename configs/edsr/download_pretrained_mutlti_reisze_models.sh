#!/bin/bash
mkdir -p pretrained_weights/edsr
for SCALE in 2 3 4
do
  wget https://github.com/quoccuongLE/mmagic/releases/download/model-v0.1.0/edsr_x${SCALE}c64b16_1xb16-300k_div2k_multi_interp_best.pth \
  -O pretrained_weights/edsr/edsr_x${SCALE}c64b16_1xb16-300k_div2k_multi_interp_best.pth
done
