#! /bin/sh

python3 test_factorvae.py --dataset dsprites --num_workers 4 --batch_size 64 \
               --output_save True --viz_on False \
               --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 \
               --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
	       --lambdaa 1.0 --ckpt_dir trained_experi_vanilla_factorvae \
               --name tmp/ --z_dim 10 --gamma 10 --ckpt_load last
