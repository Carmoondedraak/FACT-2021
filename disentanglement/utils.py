"""utils.py"""

import os
import argparse
import subprocess


class DataGather(object):
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg:[] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Definition of the FactorVAE abstract class
from abc import ABC, abstractmethod

import os
import visdom
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from .ops import recon_loss, kl_divergence, permute_dims
from .model import FactorVAE1, FactorVAE2, Discriminator
from .dataset import return_data

import numpy as np
import json


class BaseFactorVae(ABC):
    def __init__(self):
        # Misc
        self.device = None
        self.name = None
        self.max_iter = None
        self.print_iter = None
        self.global_iter = None
        self.pbar = None

        # Data
        self.dset_dir = None
        self.dataset = None
        self.batch_size = None
        self.data_loader = None
        self.data = None

        # Networks & Optimizers
        self.z_dim = None
        self.gamma = None

        self.lr_VAE = None
        self.beta1_VAE = None
        self.beta2_VAE = None

        self.lr_D = None
        self.beta1_D = None
        self.beta2_D = None

    def train(self):
        raise NotImplementedError()

    def generate_training_batch(self):
        #TODO returns a numpy ndarray
        raise NotImplementedError()
    
    def visualize_recon(self):
        data = self.image_gather.data
        true_image = data['true'][0]
        recon_image = data['recon'][0]

        true_image = make_grid(true_image)
        recon_image = make_grid(recon_image)
        sample = torch.stack([true_image, recon_image], dim=0)
        self.viz.images(sample, env=self.name+'/recon_image',
                        opts=dict(title=str(self.global_iter)))

    def visualize_line(self):
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        recon = torch.Tensor(data['recon'])
        kld = torch.Tensor(data['kld'])
        D_acc = torch.Tensor(data['acc'])
        soft_D_z = torch.Tensor(data['soft_D_z'])
        soft_D_z_pperm = torch.Tensor(data['soft_D_z_pperm'])
        soft_D_zs = torch.stack([soft_D_z, soft_D_z_pperm], -1)

        self.viz.line(X=iters,
                      Y=soft_D_zs,
                      env=self.name+'/lines',
                      win=self.win_id['D_z'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))
        self.viz.line(X=iters,
                      Y=recon,
                      env=self.name+'/lines',
                      win=self.win_id['recon'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=iters,
                      Y=D_acc,
                      env=self.name+'/lines',
                      win=self.win_id['acc'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))
        self.viz.line(X=iters,
                      Y=kld,
                      env=self.name+'/lines',
                      win=self.win_id['kld'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

    def visualize_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)

        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-limit, limit+0.1, inter)

        random_img = self.data_loader.dataset.__getitem__(0)[1]
        random_img = random_img.to(self.device).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        if self.dataset.lower() == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}

        elif self.dataset.lower() == 'celeba':
            fixed_idx1 = 191281 # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 143307 # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 101535 # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 70059  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'fixed_4':fixed_img_z4,
                 'random':random_img_z}

        elif self.dataset.lower() == '3dchairs':
            fixed_idx1 = 40919 # 3DChairs/images/4682_image_052_p030_t232_r096.png
            fixed_idx2 = 5172  # 3DChairs/images/14657_image_020_p020_t232_r096.png
            fixed_idx3 = 22330 # 3DChairs/images/30099_image_052_p030_t232_r096.png

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'random':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
            fixed_img = fixed_img.to(self.device).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z:
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            self.viz.images(samples, env=self.name+'/traverse',
                            opts=dict(title=title), nrow=len(interpolation))

        if self.output_save:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            mkdirs(output_dir)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(str(os.path.join(output_dir, key+'*.jpg')),
                         str(os.path.join(output_dir, key+'.gif')), delay=10)

        self.net_mode(train=True)

    def viz_init(self):
        zero_init = torch.zeros([1])
        self.viz.line(X=zero_init,
                      Y=torch.stack([zero_init, zero_init], -1),
                      env=self.name+'/lines',
                      win=self.win_id['D_z'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['recon'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['acc'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['kld'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'D':self.D.state_dict(),
                        'VAE':self.VAE.state_dict()}
        optim_states = {'optim_D':self.optim_D.state_dict(),
                        'optim_VAE':self.optim_VAE.state_dict()}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return
            ckpts = [str(ckpt.rstrip(".pth")) for ckpt in ckpts if "json" not in ckpt]
            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])+".pth"

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))

    def save_metrics(self, metrics):
        """ Receives the training list of metric dictionaries and saves it """
        save_file = os.path.join(self.ckpt_dir, "metrics.json")
        assert isinstance(metrics, list), "Unexpected type on metrics var" 

        if os.path.isfile(save_file):
            lst_dicts = json.load(open(save_file))
            for ele in metrics:
                lst_dicts.append(ele)
            json.dump(lst_dicts, open(save_file, mode='w'))
        else:
            json.dump(metrics, open(save_file, mode='w'))

    def disentanglement_metric(self):
        """
            It is based on "Disentangling by Factorising" paper.
            Moreover it was adapted from 
            https://github.com/nicolasigor/FactorVAE/blob/f27136ef944b5fded7cc49ecaeb398f6909cc312/vae_dsprites_v2.py#L377
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
    
        all_mus = []
        all_logvars = []
        code_list = []
        for fixed_k in range(num_factors):
            code_list_per_factor = []
            for _ in range(num_votes_per_factor): # Generate training examples per factor
                fixed_value = np.random.choice(num_classes[fixed_k]) 
                useful_samples_idx = np.where(factors[:, fixed_k] == fixed_value)[0]
                #print("The number of useful samples are", len(useful_samples_idx))
                random_idx = np.random.choice(useful_samples_idx, num_examples_per_vote)
                sample_imgs = self.data[random_idx]
                #print("The num of sampled images is with shape", sample_imgs[0].shape)
                # Get the models's predicitions/representations
                _, mus, logvars, _ = self.VAE(sample_imgs[0].to(self.device))
                mus = mus.detach().to(torch.device("cpu")).numpy()
                logvars = logvars.detach().to(torch.device("cpu")).numpy()
                #print(type(mus), type(logvars))
                all_mus.append(mus)
                all_logvars.append(logvars)
                code_list_per_factor.append((mus, logvars))
                del sample_imgs # To release the memory
            code_list.append(code_list_per_factor)
        
        all_mus = np.concatenate(all_mus, axis=0)
        all_logvars = np.concatenate(all_logvars, axis=0)
        
        # Computing the KL divergence wrt the prior
        emp_mean_kl = self.compute_kl_divergence_mean(all_mus, all_logvars)
        # Discard the dimensions that collapsed to the prior
        kl_tol = 1e-2
        useful_dims = np.where(emp_mean_kl > kl_tol)[0]

        if len(useful_dims) == 0:
            print("\nThere's no useful dim for ...\n")
            return 0

        # Compute scales for useful dims
        scales = np.std(all_mus[:, useful_dims], axis=0)

        print("The empirical mean for kl dimensions-wise:")
        print(np.reshape(emp_mean_kl, newshape=(-1,1)))
        print("Useful dimensions:", useful_dims, " - Total:", useful_dims.shape[0])
        print("Empirical Scales:", scales)

        # For the classifier - Same loop for remanining process
        d_values = []
        k_values = []
        for fixed_k in range(num_factors):
            for i in range(num_votes_per_factor):
                # Get previously generated codes
                codes = code_list[fixed_k][i][0]
                # Discarding non useful dimensions
                codes = codes[:, useful_dims]
                # Normalizing each dimension
                norm_codes = codes / scales
                emp_variance = np.var(norm_codes, axis=0)
                d_min_var = np.argmin(emp_variance)
                # The target index k provides one training input/output
                d_values.append(d_min_var)
                k_values.append(fixed_k)

        d_values = np.array(d_values)
        k_values = np.array(k_values)

        # Compute matrix V
        # The metric is the error rate of the classifier but the paper 
        #   provides accuracy instead (for comparision with previously proposed metric
        v_matrix = np.zeros((useful_dims.shape[0], num_factors))
        for j in range(useful_dims.shape[0]):
            for k in range(num_factors):
                v_matrix[j, k] = np.sum((d_values == j) & (k_values == k))

        print("Votes:\n", v_matrix)

        # Majority vote Classifier is C_j argmax_k V_jk
        classifier = np.argmax(v_matrix, axis=1)
        predicted_k = classifier[d_values]
        accuracy = np.sum(predicted_k == k_values) / num_votes
    
        print("The accuracy is", accuracy)
        return accuracy

    def compute_kl_divergence_mean(self, all_mus, all_logvar):
        """ It computes the KL divergence per dimension wrt the prior """
        variance = np.exp(all_logvar)
        squared_mean = np.square(all_mus)
        all_kl = 0.5 * (variance - all_logvar + squared_mean - 1)
        mean_kl = np.mean(all_kl, axis=0)
        return mean_kl
    
