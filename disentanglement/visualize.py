import argparse
import torch
import os
import numpy as np
import time

from tqdm import tqdm
import visdom
from PIL import Image

from utils import str2bool, mkdirs, DataGather, BaseFactorVae
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from model import FactorVAE1_Visual, FactorVAE2, Discriminator
from dataset import return_data
from ops import get_cam


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
            self.VAE = FactorVAE1_Visual(0, self.z_dim).to(self.device)
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
            self.load_checkpoint(args.ckpt_load)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        if self.output_save:
            mkdirs(self.output_dir)

    def test(self, batch_size):
        """ Generates and saves the attention maps with batch_size samples """
        fixed_idxs = [87040, 332800, 578560]

        idxs = [ fixed_idxs[i%len(fixed_idxs)] for i in range(batch_size)]
        batch = self.data[idxs]

        x_rec, (f_M,s_M), (f_c,s_c) = self.generate_attention_maps(batch)
        x_rec = x_rec.detach().cpu()
        f_M, colormaps_f = f_M.detach().cpu(), f_c.detach().cpu()
        s_M, colormaps_s = s_M.detach().cpu(), s_c.detach().cpu()

        batch = make_grid(batch[0], nrow=batch_size)
        save_image(batch.float(), '{}/batch_{}_ori.png'.format(self.output_dir, batch_size))
        bnmaps = make_grid(torch.cat((f_M,s_M), 0), nrow=batch_size)
        save_image(bnmaps.float(), '{}/batch_{}_bnmaps.png'.format(self.output_dir, batch_size))
        colormaps = make_grid(torch.cat((colormaps_f,colormaps_s), 0), nrow=batch_size)
        save_image(colormaps.float(), '{}/batch_{}_attmaps.png'.format(self.output_dir, batch_size))

    def generate_attention_maps(self, batch):
        self.net_mode(train=False)
        self.VAE.zero_grad()
        self.D.zero_grad()

        num_classes = [6,40,32,32] # freq. of latent value factors (excluding color and shape)

        with torch.set_grad_enabled(True):
            x, _ = batch
            #x = x.unsqueeze(0)
            img_shape = x.shape[2:]
            x = x.to(self.device)

            stats = self.VAE.encode(x)
            mu = stats[:, :self.VAE.z_dim]
            logvar = stats[:, self.VAE.z_dim:]
            z = self.VAE.reparametrize(mu, logvar)

            x_rec = F.sigmoid(self.VAE.decode(z).view(x.size())).data

            score = torch.sum(mu)
            score.backward(retain_graph=True)

            # Retrieve the activations and gradients
            dz_da, A = self.VAE.get_conv_output()

        # Compute attention map M and color maps
        dz_da = dz_da / (torch.sqrt(torch.mean(torch.square(dz_da))) + 1e-5)
        alpha = F.avg_pool2d(dz_da, kernel_size=dz_da.shape[2:])
        M = alpha*A
        M = F.interpolate(M, size=img_shape, mode='bilinear', align_corners=False)
        M = torch.abs(M)

        highest_Ms, sec_highest_Ms = [], []
        # For each example in the batch
        for i in range(M.shape[0]):
            m = M[i,:,:,:]
            highest_M, second_M = None, None
            maxi = torch.zeros(1).to(self.device)
            # Pick highest response map
            for j in range(M.shape[1]):
                mk = m[j]
                tmp = mk.mean()
                if tmp > maxi:
                    print(tmp, maxi)
                    second_M = highest_M
                    highest_M = mk
                    maxi = tmp

            # Add it to a list
            highest_Ms.append(highest_M)
            sec_highest_Ms.append(second_M)

        # Convert list to tensor and generate color maps
        highest_Ms = torch.stack(highest_Ms).unsqueeze(1).to(self.device)
        colormaps1 = self.create_colormap(x, highest_Ms)
        sec_highest_Ms = torch.stack(sec_highest_Ms).unsqueeze(1).to(self.device)
        colormaps2 = self.create_colormap(x, sec_highest_Ms)

        # Zero out the gradient again
        self.VAE.zero_grad()
        self.D.zero_grad()

        return x_rec, (highest_Ms, sec_highest_Ms), (colormaps1, colormaps2)

    def create_colormap(self, x, attmaps):
        """
        Creates and returns a colormap from the attention map and original input image
            x - original input images
            attmaps - attention maps from the model inferred from the input images
        """
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

    t = Tester(args)
    t.test(3)

    print("Finished after {} seconds.".format(str(time.time() - start)))


if __name__ == '__main__':
    main()
