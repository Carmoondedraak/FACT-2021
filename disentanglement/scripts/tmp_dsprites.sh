#!/bin/bash

python3 -u /home/lgpu0149/projects/FACT-2021/disentanglement/ad_factorvae.py --dataset dsprites --num_workers 4 --batch_size 64 \
               --output_save True --viz_on False \
               --viz_ll_iter 100 --viz_la_iter 500 \
               --viz_ra_iter 1000 --viz_ta_iter 1000 \
               --ckpt_save_iter 1000 --max_iter 3e3 \
               --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 \
               --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
	       --ckpt_dir experi_attention_factorvae_3000_verify/ \
               --z_dim 10 --ckpt_load last
