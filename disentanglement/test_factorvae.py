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


cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")

### Save attention maps  ###
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
        if plot:
            plt.figure(figsize=(8,4))
            subdirs = [x[1] for x in os.walk(self.ckpt_dir)]
            subdirs = subdirs[0]
            for subdir in subdirs:
                disent_vals = []
                iters = []
                #print(subdir)
                if "la_1.0" in subdir:
                    dire = os.path.join(self.ckpt_dir,subdir)
                    #print(dire)
                    for f in os.listdir(dire): 
                        path = str(f)
                        if ".pth" in path:
                            it = int(path[0:re.search(r'\b(.pth)\b', path).start()])
                            #print(it, path)
                            iters.append(it)
                            self.load_checkpoint(os.path.join(subdir, path))
                            val = self.disentanglement_metric()
                            disent_vals.append(val)
                    #print(len(iters), len(disent_vals))
                    plt.plot(iters, disent_vals, label=str(subdir))
            plt.legend(loc = 'upper center', bbox_to_anchor=(0.5,1.05), ncol=2)
            plt.ylim([0,1.3])
            plt.savefig(self.ckpt_dir+'/disent_res_abl.png')
        else:
            analyse_train_metrics()
            print("TODO implement training metrics extractor")

    def analyse_train_metrics():
        """ It receives the path to the json file with the metrics and plots them  """
       print("TODO") 

    def disentanglement_metric(self):
        """
            It is based on "Disentangling by Factorising" paper
        """
        
        self.net_mode(train=False)

        root = os.path.join(self.dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='latin1')
        factors = torch.from_numpy(data['latents_classes'])
        factors = factors[:, 1:] # Removing the color since its always white
        num_classes = [3,6,40,32,32] # the number of latent value factors
        #print("The factors are ", type(factors), factors.shape, "with classes", len(num_classes))
        num_factors = len(num_classes)

        num_examples_per_vote = 100
        num_votes = 800
        num_votes_per_factor = num_votes// num_factors
    
        try:

            all_mus = []
            all_logvars = []
            code_list = []
            for fixed_k in range(num_factors):
                code_list_per_factor = []
                for _ in range(num_votes_per_factor):
                    fixed_value = np.random.choice(num_classes[fixed_k]) 
                    useful_samples_idx = np.where(factors[:, fixed_k] == fixed_value)[0]
                    #print("The number of useful samples are", len(useful_samples_idx))
                    random_idx = np.random.choice(useful_samples_idx, num_examples_per_vote)
                    sample_imgs = self.data[random_idx]
                    #print("The num of sampled images is with shape", sample_imgs[0].shape)
                    # Get the models's predicitions
                    _, mus, logvars, _ = self.VAE(sample_imgs[0].to(self.device))
                    mus = mus.detach().to(torch.device("cpu")).numpy()
                    logvars = logvars.detach().to(torch.device("cpu")).numpy()
                    #print(type(mus), type(logvars))
                    all_mus.append(mus)
                    all_logvars.append(logvars)
                    code_list_per_factor.append((mus, logvars))
                    del sample_imgs
                code_list.append(code_list_per_factor)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| Warning: ran out of memory')
                for p in self.VAE.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                exit(0)
            else:
                raise e
        
        all_mus = np.concatenate(all_mus, axis=0)
        all_logvars = np.concatenate(all_logvars, axis=0)
        
        mean_kl = self.compute_kl_divergence_mean(all_mus, all_logvars)
        # Discard the dimensions that collapsed to the prior
        kl_tol = 1e-2
        useful_dims = np.where(mean_kl > kl_tol)[0]

        if len(useful_dims) == 0:
            print("\nThere's no useful dim for ...\n")
            return 0

        # Compute scales for useful dims
        scales = np.std(all_mus[:, useful_dims], axis=0)

        print("The empirical mean for kl dimensions-wise:")
        print(np.reshape(mean_kl, newshape=(-1,1)))
        print("Useful dimensions:", useful_dims, " - Total:", useful_dims.shape[0])
        print("Empirical Scales:", scales)

        # For the classifier
        d_values = []
        k_values = []
        for fixed_k in range(num_factors):
            #Generate training examples for this factor
            for i in range(num_votes_per_factor):
                codes = code_list[fixed_k][i][0]
                codes = codes[:, useful_dims]
                norm_codes = codes / scales
                variance = np.var(norm_codes, axis=0)
                d_min_var = np.argmin(variance)
                d_values.append(d_min_var)
                k_values.append(fixed_k)

        d_values = np.array(d_values)
        k_values = np.array(k_values)

        v_matrix = np.zeros((useful_dims.shape[0], num_factors))
        for j in range(useful_dims.shape[0]):
            for k in range(num_factors):
                v_matrix[j, k] = np.sum((d_values == j) & (k_values == k))

        print("Votes:\n", v_matrix)

        # Majority vote is C_j argmax_k V_jk
        classifier = np.argmax(v_matrix, axis=1)
        predicted_k = classifier[d_values]
        accuracy = np.sum(predicted_k == k_values) / num_votes

        print("The accuracy is", accuracy)
        return accuracy

    def compute_kl_divergence_mean(self, all_mus, all_logvar):
        """ It computes the KL divergence per dimension wrt the prior """
        variance = np.exp(all_mus, all_logvar)
        squared_mean = np.square(all_mus)
        all_kl = 0.5 * (variance - all_logvar + squared_mean - 1)
        mean_kl = np.mean(all_kl, axis=0)
        return mean_kl

    def compute_variances(self):
        raise NotImplementedError()


    def prume_dimensions(self, variances, threshold=0.):
        """ Verifies the active factors and retrieves their indexes"""
        raise NotImplementedError()
 

    def generate_training_batch(self):
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
