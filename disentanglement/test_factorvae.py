import argparse
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from .utils import str2bool, mkdirs
import time
import json


def analyse_disentanglement_metric(json_path):
    """ It receives the path to the json file and plots the proposed metric results  """
    iters, scores = [], []
    # Adding initial measure
    iters.append(0)
    scores.append(0)
    final_recon_loss = -1

    lst = json.load(open(json_path, mode="r"))
    for di in lst:
        assert isinstance(di, dict), "Got unexpected variable type"

        if di.get("metric_score") is not None:
            #if di.get("its") <=150000:
            iters.append(di["its"])
            scores.append(di["metric_score"])
        else:
            final_recon_loss = di["recon_loss"]

    return iters, scores, final_recon_loss


def get_comparison_plot(ckpt_dir, last_scores):
    """ It aims to reproduce the results observed in Figure 8 (Just the AD-FactorVAE) """
    model1_distang = [0.69, 0.685, 0.73,0.70,0.625, 0.68]
    model1_reconErr = [20, 30,42, 58, 60, 111]
    model1_value = [1,2,4,6,8,16]

    # The values were updated from trained vanilla folder
    model2_distang = [0.804, 0.823, 0.704, 0.786, 0.762]  #[0.7, 0.75,0.77,0.78,0.825]
    model2_reconErr = [54.00, 34.18, 64.92, 40.40, 92.01]  # [37, 38,39,40,40]
    model2_value = [10,20,30,40,50]  # [100,10,20,30,40]

    model3_distang = [x[0] for x in last_scores] #[0.9,0.89,0.895, 0.91]
    model3_reconErr = [x[1] for x in last_scores] #[38, 39.5, 40,40]
    model3_value = [x[2] for x in last_scores] #[10,20,30,40]

    fig, ax = plt.subplots()

    scatter = ax.scatter(model1_reconErr, model1_distang, marker="o", c='b', label='beta VAE')
    for i, txt in enumerate(model1_value):
        ax.annotate(txt, (model1_reconErr[i], model1_distang[i]), size=12)

    scatter = ax.scatter(model2_reconErr, model2_distang, marker="o", c='g', label='factor VAE')
    for i, txt in enumerate(model2_value):
        ax.annotate(txt, (model2_reconErr[i], model2_distang[i]), size=12)

    scatter = ax.scatter(model3_reconErr, model3_distang, marker="o", c='r', label='AD factor VAE')
    for i, txt in enumerate(model3_value):
        ax.annotate(txt, (model3_reconErr[i], model3_distang[i]), size=12)

    plt.rc('axes', labelsize=8)
    ax.legend(loc="upper right" )
    ax.set_title('Reconstruction error against disentanglement metric ', size = 13)
    ax.set_xlabel('reconstruction error', size = 15)
    ax.set_ylabel('disentanglement metric', size = 15)
    plt.xlim([0, 150])
    plt.ylim([0.3, 1])
    plt.grid(color='r', linestyle=':', linewidth=0.5)
    fig.savefig(ckpt_dir+'/output'+'/figure8.png')


def plot_disentanglemet_metric(ckpt_dir, seeds, vanilla):
    """ Given the root folder it extracts and plots the disentanglement metric """
    subdirs = [x[1] for x in os.walk(ckpt_dir)]
    subdirs = subdirs[0]

    fig1 = plt.figure(figsize=(9, 4))
    ax1 = plt.subplot(1, 1, 1)
    last_scores = []
    for subdir in subdirs:
        if "seed_1" in subdir:
            idx0 = subdir.index('seed')
            aver_disent, aver_recon, iters = [], [], []
            for seed in seeds: 
                #print(subdir[:idx0]+seed)
                path = os.path.join(ckpt_dir, subdir[:idx0]+seed, "metrics.json")
                iters, disent_vals, recon_loss = analyse_disentanglement_metric(path)
                if len(aver_disent) == 0:
                    aver_disent = np.zeros_like(disent_vals)
                    aver_recon = np.zeros_like(recon_loss)
                aver_disent += disent_vals
                aver_recon += recon_loss
            idx1 = subdir.index('_ga')
            idx2 = subdir.index('_la')
            idx3 = subdir.index('_iters')
            aver_disent = aver_disent / len(seeds)
            aver_recon = aver_recon / len(seeds)
            last_scores.append((aver_disent[-1], aver_recon, int(subdir[idx1+4:idx2])))
            ax1.plot(iters, aver_disent, label=subdir[idx1+1:idx3])
        plt.ylim([0, 1.1])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        ax1.set_xlabel("iters")
        ax1.set_ylabel("disentanglement metric")
        mkdirs(os.path.join(ckpt_dir, 'output/'))
        fig1.savefig(ckpt_dir+'/output'+'/disent_res_abl.png')

    if not vanilla:
        # To plot the trade-off between disentanglement metric and reconstruction loss
        get_comparison_plot(ckpt_dir, last_scores)


def analyse_train_metrics(json_path):
    """ It receives the path to the json file with the metrics and plots them  """
    iters = []
    vae_loss = []
    D_loss = []
    recon = []
    tc = []

    lst = json.load(open(json_path, mode="r"))
    for di in lst:
        assert isinstance(di, dict), "Got unexpected variable type"

        if di.get("vae_loss") is not None:
            if di.get("its") <= 150000:
                iters.append(di["its"])
                vae_loss.append(di["vae_loss"])
                D_loss.append(di["D_loss"])
                recon.append(di["recon_loss"])
                tc.append(di["tc_loss"])

    return iters, recon, tc


def plot_training_loss(ckpt_dir, seeds, detail):
    """ Given the root folder it extracts and plots the tc_loss and recon_loss over the training """
    subdirs = [x[1] for x in os.walk(ckpt_dir)]
    subdirs = subdirs[0]

    fig2, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    for subdir in subdirs:
        if "seed_1" in subdir:
            idx0 = subdir.index('seed')
            aver_recon, aver_tc, iters = [], [], []
            for seed in seeds:
                #print(subdir[:idx0]+seed)
                path = os.path.join(ckpt_dir, subdir[:idx0]+seed, "metrics.json")
                iters, recon_loss, tc_loss = analyse_train_metrics(path)
                if len(aver_recon) == 0:
                    aver_recon = np.zeros_like(recon_loss)
                    aver_tc = np.zeros_like(tc_loss)
                aver_recon += recon_loss
                aver_tc += tc_loss
            aver_recon = aver_recon / len(seeds)
            idx1 = subdir.index('_ga')
            idx2 = subdir.index('_la')#'_iters')
            aver_tc = aver_tc / len(seeds)
            axs[0].plot(iters, aver_recon, label=subdir[idx1+1:idx2])
            axs[1].plot(iters, aver_tc, label=subdir[idx1+1:idx2])
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    if detail:
        axs[0].set_ylim([0, 150])
        axs[1].set_ylim([-0.3, 0.7])
    axs[1].set_xlabel("iters")
    axs[0].set_ylabel("recon_loss")
    axs[1].set_ylabel("tc_loss")
    mkdirs(os.path.join(ckpt_dir, 'output/'))
    fig2.savefig(ckpt_dir+'/output'+'/disent_train_metrics_abl.png')


def main():
    parser = argparse.ArgumentParser(description='Factor-VAE')

    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--gamma', default=6.4, type=float, help='gamma hyperparameter')
    parser.add_argument('--lambdaa', default=1.0, type=float, help='attention disentanglement hyperparameter')
    parser.add_argument('--lr_VAE', default=1e-4, type=float, help='learning rate of the VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type=float, help='beta2 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of the discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of the Adam optimizer for the discriminator')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CelebA', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port number')
    parser.add_argument('--viz_ll_iter', default=1000, type=int, help='visdom line data logging iter')
    parser.add_argument('--viz_la_iter', default=5000, type=int, help='visdom line data applying iter')
    parser.add_argument('--viz_ra_iter', default=10000, type=int, help='visdom recon image applying iter')
    parser.add_argument('--viz_ta_iter', default=10000, type=int, help='visdom traverse applying iter')

    parser.add_argument('--print_iter', default=500, type=int, help='print losses iter')

    parser.add_argument('--ckpt_dir', default='experiments', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_save_iter', default=10000, type=int, help='checkpoint save iter')

    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--output_save', default=True, type=str2bool, help='whether to save traverse results')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    
    args = parser.parse_args()

    # To achieve reproducible results with sequential runs
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    init_seed = args.seed
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    np.random.seed(init_seed)

    start = time.time()

    seeds = ['seed_1', 'seed_2']
    plot_training_loss(args.ckpt_dir, seeds)
    #plot_disentanglemet_metric(args.ckpt_dir, seeds, vanilla=True)

    print("Finished after {} seconds.".format(str(time.time() - start)))


if __name__ == '__main__':
    main()
