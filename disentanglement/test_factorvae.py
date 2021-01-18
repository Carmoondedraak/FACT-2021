import argparse
import torch

import os
import numpy as np

from tqdm import tqdm
import visdom
from matplotlib import pyplot as plt

from utils import str2bool
import time

import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils import DataGather, mkdirs, grid2gif, BaseFactorVae
from ops import recon_loss, kl_divergence, permute_dims, attention_disentanglement
from model import FactorVAE1, FactorVAE2, Discriminator
from dataset import return_data

import json
from gradcam import GradCamDissen

import re
from pathlib import Path



### Save attention maps  ### - TODO to use with the attetion maps
def save_cam(image, filename, gcam):
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    h, w, d = image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=np.float) + \
        np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    cv2.imwrite(filename, gcam)

### Define the FactorVAE disentanglement metric tester

class Tester(BaseFactorVae):
    def __init__(self, args):
        # Misc
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = args.name
        self.max_iter = int(args.max_iter)
        self.print_iter = args.print_iter
        self.global_iter = 0
        self.pbar = tqdm(total=self.max_iter)

        # Data
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader, self.data = return_data(args)

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.gamma = args.gamma

        self.lambdaa = args.lambdaa

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        if args.dataset == 'dsprites':
            self.VAE = FactorVAE1(self.z_dim).to(self.device)
            self.nc = 1
        else:
            self.VAE = FactorVAE2(self.z_dim).to(self.device)
            self.nc = 3
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = Discriminator(self.z_dim).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))

        self.nets = [self.VAE, self.D]
        
        # Visdom
        """
        self.viz_on = args.viz_on
        self.win_id = dict(D_z='win_D_z', recon='win_recon', kld='win_kld', acc='win_acc')
        self.line_gather = DataGather('iter', 'soft_D_z', 'soft_D_z_pperm', 'recon', 'kld', 'acc')
        self.image_gather = DataGather('true', 'recon')
        if self.viz_on:
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom(log_to_filename='./logging.log', offline=True)  # server='http://188.140.46.6' ,port=self.viz_port)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter
            self.viz_ra_iter = args.viz_ra_iter
            self.viz_ta_iter = args.viz_ta_iter
            if not self.viz.win_exists(env=self.name+'/lines', win=self.win_id['D_z']):
                self.viz_init()
        """

        # Checkpoint
        self.ckpt_dir = args.ckpt_dir
        self.ckpt_save_iter = args.ckpt_save_iter
        mkdirs(self.ckpt_dir)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        mkdirs(self.output_dir)

    def test(self, plot=True):
        subdirs = [x[1] for x in os.walk(self.ckpt_dir)]
        subdirs = subdirs[0]
        if plot:
            fig = plt.figure(figsize=(8,4))
            ax = plt.subplot(1,1,1)
            for subdir in subdirs:
                disent_vals = []
                iters = []
                #print(subdir)
                if "la_0.33" in subdir:
                    path = os.path.join(self.ckpt_dir,subdir, "metrics.json")
                    iters, disent_vals = self.analyse_disentanglement_metric(path)
                    ax.plot(iters, disent_vals, label=str(subdir))
            ax.legend(loc = 'upper center', bbox_to_anchor=(0.5,1.05), ncol=2)
            ax.set_xlabel("iters")
            ax.set_ylabel("recon_loss")
            mkdirs(os.path.join(self.ckpt_dir,'output/'))
            plt.savefig(self.ckpt_dir+'/output'+'/disent_res_abl.png')
        else:
            fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout = True)
            for subdir in subdirs:
                #print(subdir)
                if "la_0.33" in subdir:
                    path = os.path.join(self.ckpt_dir,subdir, "metrics.json")
                    iters, recon_loss, tc_loss = self.analyse_train_metrics(path)
                    axs[0].plot(iters, recon_loss)
                    axs[1].plot(iters, tc_loss)

            axs[1].set_xlabel("iters")
            axs[0].set_ylabel("recon_loss")
            axs[1].set_ylabel("tc_loss")
            mkdirs(os.path.join(self.ckpt_dir,'output/'))
            plt.savefig(self.ckpt_dir+'/output'+'/disent_train_metrics_abl.png')

    def analyse_train_metrics(self, json_path):
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
                iters.append(di.get("its"))
                vae_loss.append(di["vae_loss"])
                D_loss.append(di["D_loss"])
                recon.append(di["recon_loss"])
                tc.append(di["tc_loss"])
        
        return iters, recon, tc     

    def analyse_disentanglement_metric(self, json_path):
        """ It receives the path to the json file and plots the proposed metric results  """
        iters = []
        scores = []

        lst = json.load(open(json_path, mode="r"))
        for di in lst:
            assert isinstance(di, dict), "Got unexpected variable type"

            if di.get("metric_score") is not None:
                iters.append(di.get("its"))
                scores.append(di["metric_score"])
        
        return iters, scores     

    def produce_comparison_plot(self):
        """ It aims to reproduce the results observed in Figure 8 (Just the AD-FactorVAE) """
        #TODO
        raise NotImplementedError()


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

    
    tester = Tester(args)
    start = time.time()
    tester.test()
    print("Finished after {} mins.".format(str((time.time() - start) // 60)))

if __name__ == '__main__':
    main()
