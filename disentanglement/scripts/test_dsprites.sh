#! /bin/sh

python3 test_factorvae.py --dataset dsprites --num_workers 4 --batch_size 64 \
               --output_save True --viz_on False \
               --ckpt_save_iter 1000 --max_iter 6e3 \
               --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 \
               --lr_D 1e-4 --beta1_D 0.5 --beta2_D 0.9 \
	       --lambdaa 1.0 --ckpt_dir experiments_new \
               --name test/ --z_dim 10 --gamma 10 --ckpt_load last
