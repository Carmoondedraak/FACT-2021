#! /bin/sh

python3 train_FactorVAE.py --dataset dsprites --num_workers 4 --batch_size 64 \
               --output_save True --viz_on True \
               --viz_ll_iter 100 --viz_la_iter 500 \
               --viz_ra_iter 1000 --viz_ta_iter 1000 \
               --ckpt_save_iter 1000 --max_iter 6e3 \
               --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 \
               --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
	       --lambdaa 1.0 \
               --name $1 --z_dim 10 --gamma 10 --ckpt_load last
