"""solver.py"""

import os
import visdom
from tqdm import tqdm

import argparse
import numpy as np
from utils import str2bool
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils import DataGather, mkdirs, grid2gif, BaseFactorVae
from ops import recon_loss, kl_divergence, permute_dims, attention_disentanglement, GradCamDissen
from model import FactorVAE1, FactorVAE2, Discriminator
from dataset import return_data

import json


class Solver(BaseFactorVae):
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

        self.lambdaa = args.lambdaa # TODO analyse the effect of this parameter

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        if args.dataset == 'dsprites':
            self.VAE = FactorVAE1(self.z_dim).to(self.device)
            self.nc = 1
            """
            for m in self.VAE.named_modules():
                print(m[0])
            """
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

        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        self.ckpt_save_iter = args.ckpt_save_iter
        mkdirs(self.ckpt_dir)
        if args.ckpt_load:
            self.load_checkpoint(args.ckpt_load)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        mkdirs(self.output_dir)

    def select_attention_maps(self, maps):
        """
            Implements the mechanism of selection of the attention
                maps to use in the attention disentanglement loss
        """
        #print(type(maps[0]), type(maps[0].detach().to(torch.device("cpu"))))
        pairs = [(maps[i], maps[i+1]) for i in range(len(maps[:-1]))]
        return pairs

    def train(self):

        gcam = GradCamDissen(self.VAE, self.D, target_layer='encode.10', cuda=True) # The FactorVAE encoder contains 6 layers
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        mu_avg, logvar_avg, test_index = 0, 1, 0
        metrics = []
        out = False
        while not out:
            for batch_idx, (x1, x2) in enumerate(self.data_loader):
                self.global_iter += 1
                self.pbar.update(1)

                x1 = x1.to(self.device)
                x1_rec, mu, logvar, z = gcam.forward(x1)
                # For Standard FactorVAE loss
                vae_recon_loss = recon_loss(x1, x1_rec)
                vae_kld = kl_divergence(mu, logvar)
                D_z = self.D(z)
                vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean() # TODO where is the log in the equation ??

                factorVae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss
                # For attention disentanglement loss
                gcam.backward(mu, logvar, mu_avg, logvar_avg)
                att_loss = 0
                with torch.no_grad():
                    gcam_maps = gcam.generate()
                    selected = self.select_attention_maps(gcam_maps)
                    for (sel1, sel2) in selected:
                        att_loss += attention_disentanglement(sel1, sel2)
                #print("The total att_loss is {}".format(att_loss))
                att_loss /= len(selected) # Averaging the loss accross all pairs of maps
                #print("The average att_loss is {}".format(att_loss))

                vae_loss = factorVae_loss + self.lambdaa*att_loss
                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()

                x2 = x2.to(self.device)
                z_prime = self.VAE(x2, no_dec=True)
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.D(z_pperm)
                D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_D.step()

                # Saving the training metrics
                if self.global_iter % 100 == 0:
                    metrics.append({'its':self.global_iter, 'vae_loss': vae_loss.detach().to(torch.device("cpu")).item(), 'D_loss': D_tc_loss.detach().to(torch.device("cpu")).item(), 'recon_loss':vae_recon_loss.detach().to(torch.device("cpu")).item(), 'tc_loss': vae_tc_loss.detach().to(torch.device("cpu")).item()})

                # Saving the disentanglement metrics results
                if self.global_iter % 1500 == 0:
                    score = self.disentanglement_metric() 
                    metrics.append({'its':self.global_iter, 'metric_score': score})
                    self.net_mode(train=True) #To continue the training again

                if self.global_iter%self.print_iter == 0:
                    self.pbar.write('[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                        self.global_iter, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item()))
                
                if self.global_iter%self.ckpt_save_iter == 0:
                    self.save_checkpoint(str(self.global_iter)+".pth")
                    self.save_metrics(metrics)
                    metrics = []
                """
                ### For writing the generated images organized in grids 
                if self.viz_on and (self.global_iter%self.viz_ll_iter == 0):
                    soft_D_z = F.softmax(D_z, 1)[:, :1].detach()
                    soft_D_z_pperm = F.softmax(D_z_pperm, 1)[:, :1].detach()
                    D_acc = ((soft_D_z >= 0.5).sum() + (soft_D_z_pperm < 0.5).sum()).float()
                    D_acc /= 2*self.batch_size
                    self.line_gather.insert(iter=self.global_iter,
                                            soft_D_z=soft_D_z.mean().item(),
                                            soft_D_z_pperm=soft_D_z_pperm.mean().item(),
                                            recon=vae_recon_loss.item(),
                                            kld=vae_kld.item(),
                                            acc=D_acc.item())

                if self.viz_on and (self.global_iter%self.viz_la_iter == 0):
                    self.visualize_line()
                    self.line_gather.flush()

                if self.viz_on and (self.global_iter%self.viz_ra_iter == 0):
                    self.image_gather.insert(true=x_true1.data.cpu(),
                                             recon=F.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    self.image_gather.flush()

                if self.viz_on and (self.global_iter%self.viz_ta_iter == 0):
                    if self.dataset.lower() == '3dchairs':
                        self.visualize_traverse(limit=2, inter=0.5)
                    else:
                        self.visualize_traverse(limit=3, inter=2/3)
                """               
                """
                ## Visualize and save attention maps  ##
                x1 = x.repeat(1, 3, 1, 1)
                for i in range(x1.size(0)):
                    raw_image = x1[i] * 255.0
                    ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()
                    im = Image.fromarray(ndarr.astype(np.uint8))
                    im_path = args.result_dir
                    if not os.path.exists(im_path):
                        os.mkdir(im_path)
                    im.save(os.path.join(im_path,
                                     "{}-{}-origin.png".format(test_index, str(one_class))))

                    file_path = os.path.join(im_path,
                                         "{}-{}-attmap.png".format(test_index, str(one_class)))
                    r_im = np.asarray(im)
                    save_cam(r_im, file_path, gcam_map[i].squeeze().cpu().data.numpy())
                    test_index += 1

                """
                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Training Finished]")
        self.pbar.close()


def experiment1(args, seed):
    """ Ablation study comparing Vanilla FactorVAE
            with AD-FactorVAE
    """
    gammas = [10, 20, 30, 40]
    lambdas = [1.0]
    for la in lambdas:
        args.lambdaa = la
        for ga in gammas:
            args.gamma = ga
            args.name = "disent_ga_{}_la_{}_iters_{}_seed_{}/".format(args.gamma, args.lambdaa, int(args.max_iter), seed)
            solver = Solver(args)
            solver.train()
            del solver


def experiment2(args, seed):
    """ Ablation study comparing the influence of gamma
            hyperparameter on AD-FactorVAE performance
    """
    gammas = [5, 15, 25, 35]
    lambdas = [1.0]
    for la in lambdas:
        args.lambdaa = la
        for ga in gammas:
            args.gamma = ga
            args.name = "disent_ga_{}_la_{}_iters_{}_seed_{}/".format(args.gamma, args.lambdaa, int(args.max_iter), seed)
            solver = Solver(args)
            solver.train()
            del solver


def experiment3(args, seed):
    """ Ablation study comparing the influence of lambda
            hyperparameter on AD-FactorVAE performance
    """
    gammas = [10, 20, 30, 40]
    lambdas = [0.5, 1.5]
    for la in lambdas:
        args.lambdaa = la
        for ga in gammas:
            args.gamma = ga
            args.name = "disent_ga_{}_la_{}_iters_{}_seed_{}/".format(args.gamma, args.lambdaa, int(args.max_iter), seed)
            solver = Solver(args)
            solver.train()
            del solver


if __name__ == "__main__":
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

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_save_iter', default=10000, type=int, help='checkpoint save iter')

    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--output_save', default=True, type=str2bool, help='whether to save traverse results')

    args = parser.parse_args()

    seeds = [1]
    start = time.time()
    for seed in seeds:
        # To achieve reproducible results with sequential runs
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        init_seed = seed
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed(init_seed)
        np.random.seed(init_seed)

        experiment1(args, seed)

    print("Finished after {} mins.".format(str((time.time() - start) // 60)))
