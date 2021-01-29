#!/bin/bash

python vanilla.py --dataset dsprites --num_workers 4 --batch_size 64 \
               --output_save True --viz_on False \
               --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 \
               --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
	       --ckpt_dir experi_vanilla_factorvae \
               --z_dim 10 --ckpt_load last
