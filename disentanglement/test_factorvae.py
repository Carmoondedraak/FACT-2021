import argparse
import torch

import os
import numpy as np

from matplotlib import pyplot as plt

from utils import str2bool
import time

from utils import mkdirs

import json

from tqdm import tqdm
import visdom

import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils import DataGather, BaseFactorVae
from ops import recon_loss, kl_divergence, permute_dims, attention_disentanglement, get_cam, GradCamDissen
from model import FactorVAE1, FactorVAE_Dsprites, FactorVAE2, Discriminator
from dataset import return_data

from PIL import Image


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
            self.VAE = FactorVAE_Dsprites(10, self.z_dim).to(self.device)
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
        self.viz_on = args.viz_on
        self.win_id = dict(D_z='win_D_z', recon='win_recon', kld='win_kld', acc='win_acc')
        self.line_gather = DataGather('iter', 'soft_D_z', 'soft_D_z_pperm', 'recon', 'kld', 'acc')
        self.image_gather = DataGather('true', 'recon')
        if self.viz_on:
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom(log_to_filename='./logging.log', offline=True)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter
            self.viz_ra_iter = args.viz_ra_iter
            self.viz_ta_iter = args.viz_ta_iter
            if not self.viz.win_exists(env=self.name+'/lines', win=self.win_id['D_z']):
                self.viz_init()

        # Checkpoint
        self.ckpt_dir = args.ckpt_dir
        self.ckpt_save_iter = args.ckpt_save_iter
        mkdirs(self.ckpt_dir)
        if args.ckpt_load:
            #args.ckpt_load = '1000.pth'
            self.load_checkpoint(args.ckpt_load)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        if self.output_save:
            mkdirs(self.output_dir)

    def test(self, batch_size, shape):
        fixed_idx = {'square': (87040, 332800),
                    'ellipse': (332800, 578560),
                    'heart': (578560, 737280)}[shape]

        s, e  = fixed_idx
        useful_samples_idx = torch.tensor([i for i in range(s, e, 1)], dtype=torch.float)
        random_idx = torch.multinomial(useful_samples_idx, batch_size)
        batch = self.data[random_idx]

        x_rec, M, colormaps = self.generate_attention_maps(batch)
        x_rec, M, colormaps = x_rec.detach().cpu(), M.detach().cpu(), colormaps.detach().cpu()

        batch = make_grid(batch[0], nrow=batch_size)
        save_image(batch.float(), '{}/batch_{}_ori.png'.format(self.output_dir, batch_size))
        bnmaps = make_grid(M, nrow=batch_size)
        save_image(bnmaps.float(), '{}/batch_{}_bnmaps.png'.format(self.output_dir, batch_size))
        colormaps = make_grid(colormaps, nrow=batch_size)
        save_image(colormaps.float(), '{}/batch_{}_attmaps.png'.format(self.output_dir, batch_size))

    def generate_attention_maps(self, batch, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        self.VAE.zero_grad()
        self.D.zero_grad()

        with torch.set_grad_enabled(True):
            x, _ = batch
            #x = x.unsqueeze(0)
            img_shape = x.shape[2:]
            x = x.to(self.device) 
            x_rec, mu, logvar, z = self.VAE(x)
            #n_channels = x.shape[1]

            score = torch.sum(mu)
            score.backward(retain_graph=True)

            #Retrieve the activations and gradients
            dz_da, A = self.VAE.get_conv_output()

        # Compute attention map M and color maps
        dz_da = dz_da / (torch.sqrt(torch.mean(torch.square(dz_da))) + 1e-5)
        alpha = F.avg_pool2d(dz_da, kernel_size=dz_da.shape[2:])
        A, alpha = A, alpha

        A, alpha = A.unsqueeze(0), alpha.unsqueeze(1)
        M = F.conv3d(A, (alpha), padding=0, groups=len(alpha)).squeeze(0).squeeze(1)
        M = F.interpolate(M.unsqueeze(1), size=img_shape, mode='bilinear', align_corners=False)
        M = torch.abs(M)

        highest_M, maxi = None, torch.zeros(1).to(self.device)
        for m in M:
            mapp = m.squeeze(0)
            tmp = mapp.mean()
            if tmp > maxi:
                print(tmp, maxi)
                highest_M = mapp
                maxi = tmp

        colormaps = self.create_colormap(x, M) #highest_M)

        # Zero out the gradient again
        self.VAE.zero_grad()
        self.D.zero_grad()

        return x_rec, M, colormaps

    """
    def unnormalize_batch(self, imgs, mean, std, n_channels):
        mean = torch.tensor(mean).view(n_channels, 1, 1)
        std = torch.tensor(std).view(n_channels, 1, 1)
        un_tensor = imgs * std.to(self.device)
        un_tensor = un_tensor + mean.to(self.device)
        return un_tensor
    """

    def create_colormap(self, x, attmaps, unnormalize=False):
        """
        Creates and returns a colormap from the attention map and original input image
            x - original input images
            attmaps - attention maps from the model inferred from the input images
        """
        if unnormalize:
            x = self.unnormalize_batch(x)
        attmaps = attmaps.detach()
        n_channels = x.shape[1]
        if n_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        colormaps = torch.zeros(x.shape)
        for i in range(x.size(0)):
            raw_image = x[i] * 255.0
            ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()
            im = Image.fromarray(ndarr.astype(np.uint8))

            r_im = np.asarray(im)
            gcam = get_cam(r_im, attmaps[i].squeeze().cpu().data.numpy())
            colormaps[i] = torch.from_numpy(gcam).permute(2, 0, 1)/255

        permute = [2, 1, 0]
        colormaps = colormaps[:, permute]
        return colormaps


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
            #if di.get("its") <= 150000:
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
    # 300k [0.804, 0.823, 0.704, 0.786, 0.762]  # figure [0.7, 0.75,0.77,0.78,0.825]
    model2_distang = [0.724, 0.826, 0.721, 0.778, 0.756] 
    #300k [54.00, 34.18, 64.92, 40.40, 92.01]  #figure [37, 38,39,40,40]
    model2_reconErr = [54.001, 34.181, 64.920, 40.402, 92.014] 
    #figure [100,10,20,30,40]
    model2_value = [10,20,30,40,50]

    """
    print("The vanilla results are")
    print([x[0] for x in last_scores])
    print([x[1] for x in last_scores])
    print([x[2] for x in last_scores])
    """

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


def plot_disentanglemet_metric(ckpt_dir, seeds):
    """ Given the root folder it extracts and plots the disentanglement metric """
    subdirs = [x[1] for x in os.walk(ckpt_dir)]
    subdirs = subdirs[0]

    fig1 = plt.figure(figsize=(9, 4))
    ax1 = plt.subplot(1, 1, 1)
    last_scores = []
    for subdir in subdirs:
        if "seed_1" in subdir and ("ga_10" in subdir or "ga_20" in subdir or "ga_30" in subdir or "ga_40" in subdir  or "ga_50" in subdir):
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
            p = ax1.plot(iters, aver_disent, label=subdir[idx1+1:idx3])
            """
            #Get index 150000
            i = iters.index(150000)
            ax1.axhline(aver_disent[i], color=p[-1].get_color())
            """
        plt.ylim([0, 1.1])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        ax1.set_xlabel("iters")
        ax1.set_ylabel("disentanglement metric")
        mkdirs(os.path.join(ckpt_dir, 'output/'))
        fig1.savefig(ckpt_dir+'/output'+'/disent_res_abl.png')

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
            iters.append(di.get("its"))
            vae_loss.append(di["vae_loss"])
            D_loss.append(di["D_loss"])
            recon.append(di["recon_loss"])
            tc.append(di["tc_loss"])

    return iters, recon, tc


def plot_training_loss(ckpt_dir, seeds):
    """ Given the root folder it extracts and plots the tc_loss and recon_loss over the training """
    subdirs = [x[1] for x in os.walk(ckpt_dir)]
    subdirs = subdirs[0]

    fig2, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    for subdir in subdirs:
        if "seed_1" in subdir and ("ga_10" in subdir or "ga_20" in subdir or "ga_30" in subdir or "ga_40" in subdir  or "ga_50" in subdir):
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
            idx2 = subdir.index('_iters')
            aver_tc = aver_tc / len(seeds)
            axs[0].plot(iters, aver_recon, label=subdir[idx1+1:idx2])
            axs[1].plot(iters, aver_tc, label=subdir[idx1+1:idx2])
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    #axs[0].set_ylim([0, 150])
    #axs[1].set_ylim([-0.3, 0.7])
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
    parser.add_argument('--dataset', default='dsprites', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port number')
    parser.add_argument('--viz_ll_iter', default=1000, type=int, help='visdom line data logging iter')
    parser.add_argument('--viz_la_iter', default=5000, type=int, help='visdom line data applying iter')
    parser.add_argument('--viz_ra_iter', default=10000, type=int, help='visdom recon image applying iter')
    parser.add_argument('--viz_ta_iter', default=10000, type=int, help='visdom traverse applying iter')

    parser.add_argument('--print_iter', default=500, type=int, help='print losses iter')

    parser.add_argument('--ckpt_dir', default='', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default='100000.pth', type=str, help='checkpoint name to load')
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
    #plot_training_loss(args.ckpt_dir, seeds)
    #plot_disentanglemet_metric(args.ckpt_dir, seeds)
    t = Tester(args)
    #t.generate_figure_rows('heart', n_samples=10, limit=3, inter=2/3)
    t.test(1, 'heart')
    #t.generate_attention_maps()

    print("Finished after {} seconds.".format(str(time.time() - start)))


if __name__ == '__main__':
    main()
