import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import argparse
from PIL import Image
from utils import *
from MNIST_Dataset import OneClassMNISTDataModule
from UCSD_Dataset import UCSDDataModule
from UCSD_Dataset import UCSD
from torch.utils.data import DataLoader
from vae_model import VAE
from torchvision.utils import make_grid, save_image
from torchvision.datasets import MNIST
from torchvision import transforms

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

class ExpVAE(pl.LightningModule):
    
    def __init__(self, im_shape, lr=1e-3, inference_mode='mean_sum', layer_idx=2, z_dim=32):
        super().__init__()

        self.save_hyperparameters()
        
        self.vae = VAE(self.hparams.layer_idx, self.hparams.z_dim, self.hparams.im_shape)


    def loss_f(self, recon_x, x, mu, logvar):
        """
        Function which calculates the VAE loss using BCE and KL divergence
            x - that original "target" images
            x_rec - reconstructed images from the model
            mu, log_var - mean and log variance output of the encoder for calculating KL term
        """

        batch_size = x.shape[0]

        # Reconstruction loss using BCE
        # L_rec = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none').mean()
        L_rec = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')

        # KL divergence between encoder and unit Gaussian
        L_reg = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Compute final loss
        loss = L_rec - L_reg
        return loss

    def training_step(self, batch, batch_idx):
        """
        Defines a single training iteration with the goal of reconstructing the input
        """
        x,  _ = batch

        x_rec, mu, log_var = self.vae(x)

        loss = self.loss_f(x_rec, x, mu, log_var)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines a single validation iteration with the goal of reconstructing the input
        """
        x,  _ = batch

        x_rec, mu, log_var = self.vae(x)

        loss = self.loss_f(x_rec, x, mu, log_var)

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines a single testing iteration with the goal of reconstructing the input
        """
        x,  _ = batch

        x_rec, mu, log_var = self.vae(x)

        loss = self.loss_f(x_rec, x, mu, log_var)

        self.log('test_loss', loss)

        # TODO: Doesn't work right now because testing doesn't include grads, but self.forward requires grads
        # x_rec, M, colormaps = self.forward(x)

        # save_image(make_grid(colormaps.float()), os.path.join(self.logger.log_dir, 'attmaps_test_sample.png'))

        return loss

    def create_colormap(self, x, attmaps):
        """
        Creates and returns a colormap from the attention map and original input image
            x - original input images
            attmaps - attention maps from the model inferred from the input images
        """
        attmaps = attmaps.detach()
        x = x.repeat(1, 3, 1, 1)
        colormaps = torch.zeros(x.shape)
        for i in range(x.size(0)):
            raw_image = x[i] * 255.0
            ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()
            im = Image.fromarray(ndarr.astype(np.uint8))

            r_im = np.asarray(im)
            gcam = get_cam(r_im, attmaps[i].squeeze().cpu().data.numpy())
            colormaps[i] = torch.from_numpy(gcam).permute(2,0,1)/255

        permute = [2,1,0]
        colormaps = colormaps[:, permute]
        return colormaps

    def forward(self, x):
        """
        Forward function which reconstructs input, and also returns attention and colormaps
        Note that this function is not used during training, only for evaluation
        """
        self.eval()
        x = x.to(self.device)
        x_rec, mu, log_var = self.vae(x)
        x_rec = torch.sigmoid(x_rec)
        
        if self.hparams.inference_mode == 'mean_sum':
            score = torch.sum(mu)
        elif self.hparams.inference_mode == 'normal_diff':
            z = self.vae.norm_diff_reparametrize(mu, log_var)
            score = torch.sum(z)

        self.zero_grad()
        score.backward(retain_graph=True)
        dz_da, A = self.vae.get_layer_data()
        
        with torch.no_grad():
            dz_da = dz_da / (torch.sqrt(torch.mean(torch.square(dz_da))))
            alpha = F.avg_pool2d(dz_da, kernel_size=dz_da.shape[2:])
            M = torch.sum(alpha*A, dim=1)
            M = F.interpolate(M.unsqueeze(0), size=self.hparams.im_shape[1:], mode='bilinear', align_corners=False).permute(1,0,2,3)
            M = torch.abs(M)

        colormaps = self.create_colormap(x, M)

        return x_rec, M, colormaps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def set_normal_stats(self, mu, log_var):
        self.vae.configure_normal(mu=mu, log_var=log_var)
        self.hparams.inference_mode = 'normal_diff'


class SampleAttentionCallback(pl.Callback):

    def __init__(self, batch_size, every_n_epoch=5):
        """
        PyTorch Lightning callback which handles saving samples throughout training
        Inputs:
            batch_size - number of images to use for sampling/generation
            every_n_epochs - only save images every n epochs
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epoch = every_n_epoch
        self.epoch = 0
    
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """
        Function called every training epoch, calls generate_samples every n epochs
        """
        if (self.epoch) % self.every_n_epoch == 0:
            self.generate_samples(trainer, pl_module)
        self.epoch += 1

    def sample_images(self, trainer, pl_module, oc=True, mode='attmaps', include_orig=False):
        """
        Function which either saves attention maps, reconstructed images and saves original images
            trainer - PyTorch Lightning trainer object, through which we access the necessary dataloaders
            pl_module - PyTorch Lightning module used for inference
            oc - Boolean switch representing "other class". When True, does inference on images which the 
                model was not trained on
            mode - Specifies whether we want the generated attention maps or reconstructed images
                "attmaps" or "rec"
            inculde_orig - Boolean switch on whether we want to save the original input images as well
        """
        # Get dataset name
        dataset_name = pl_module.train_dataloader().dataset.__class__.__name__

        # Get the correct dataset for inference
        if oc:
            if 'mnist' in dataset_name.lower():
                imgs, _ = next(iter(pl_module.val_dataloader()))
                digit = pl_module.val_dataloader().dataset.digit
            elif 'ucsd' in dataset_name.lower():
                imgs, target = next(iter(pl_module.val_dataloader()))
                digit = 10
                save_image(make_grid(target), f'{trainer.logger.log_dir}/{self.epoch}-targets.png')
        else:
            if 'mnist' in dataset_name.lower():
                imgs, _ = next(iter(trainer.train_dataloader))
                digit = trainer.train_dataloader.dataset.digit
            elif 'ucsd' in dataset_name.lower():
                imgs, _ = next(iter(trainer.train_dataloader))
                digit = 10

        # Save attention maps or reconstructed images
        if mode == 'attmaps':
            _, _, colormaps = pl_module.forward(imgs)
            colormaps_grid = make_grid(colormaps)
            save_image(colormaps_grid.float(), f'{trainer.logger.log_dir}/{self.epoch}-{digit}-attmaps.png')
        elif mode == 'rec':
            imgs_rec, _, _ = pl_module.forward(imgs)
            imgs_rec, _ = imgs_rec.detach(), _.detach()
            imgs_rec_grid = make_grid(imgs_rec)
            save_image(imgs_rec_grid.float(), f'{trainer.logger.log_dir}/{self.epoch}-{digit}-rec.png')

        # Save original image if include_orig == True
        if include_orig:
            imgs_grid = make_grid(imgs)
            save_image(imgs_grid.float(), f'{trainer.logger.log_dir}/{self.epoch}-{digit}-orig.png')

    def generate_samples(self, trainer, pl_module):
        # Generate reconstructed images to see if model is training
        self.sample_images(trainer, pl_module,oc=False, mode='rec')
        # Generate attentionmaps from "other class" images as well
        self.sample_images(trainer, pl_module, oc=True, mode='attmaps', include_orig=True)

def exp_vae(args):
    set_work_directory()

    # First pick the correct dataset
    if args.dataset.lower() == 'mnist':
        log_dir = 'mnist_logs'
        dm = OneClassMNISTDataModule(root='./Datasets/MNIST_dataset')
    elif args.dataset.lower() == 'ucsd':
        log_dir = 'ucsd_logs'
        dm = UCSDDataModule(root='./Datasets/UCSD_dataset')
    elif args.dataset.lower() == 'mvtec':
        log_dir = 'mvtec_logs'
        raise NotImplementedError
    elif args.dataset.lower() == 'dsprites':
        log_dir = 'dsprites_logs'
        raise NotImplementedError


    # Create PyTorch Lightning trainer with callback for sampling
    if args.sample_during_training:
        att_map_cb = SampleAttentionCallback(batch_size=args.batch_size, every_n_epoch=args.sample_every_n_epoch)
    else:
        att_map_cb = None

    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
    )
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        gpus=1 if torch.cuda.is_available() else 0, 
        checkpoint_callback=checkpoint_cb,
        callbacks=[att_map_cb], 
        max_epochs=args.epochs,
        progress_bar_refresh_rate=1 if args.progress_bar else 0
    )

    # Choosing which inferencing method to use
    if args.inference_mode == 'normal_diff':
        mu, var = calc_latent_mu_var(model.vae, dm, args.batch_size)
        model.set_normal_stats(mu, var)

    # Either train or test
    if args.eval:
        model_path, hparams_path = get_ckpt_path(log_dir, args)

        model = ExpVAE.load_from_checkpoint(
            checkpoint_path=model_path,
            hparams_file=hparams_path
            )

        # Make sure dataset is prepared/downloaded
        dm.prepare_data()
        dm.setup('test')
        
        trainer.test(model, datamodule=dm)
    else:
        # Make sure dataset is prepared/downloaded
        dm.prepare_data()
        dm.setup()

        # Load pretrained model if it exists
        if args.model_version is not None:
            model_path, hparams_path = get_ckpt_path(log_dir, args)

            model = ExpVAE.load_from_checkpoint(
                checkpoint_path=model_path,
                hparams_file=hparams_path
                )
        else:
            # Initialize the model
            im_size = dm.dims
            model = ExpVAE(im_size, layer_idx=args.layer_idx)

        trainer.fit(model, dm)

torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Optimizer and Training Hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for training')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to use for training')
    parser.add_argument('--progress_bar', default=True, type=bool, help='Show or hide progress bar during training')

    # Model Hyperparameters
    parser.add_argument('--layer_idx', default=2, type=int, help='Layer number to use for attention map generation')
    parser.add_argument('--z_dim', default=32, type=int, help='Latent dimension size for the VAE')

    # Dataset options
    parser.add_argument('--dataset', default='mnist', type=str, help='Dataset used for training and visualization')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
    
    # Inference option
    parser.add_argument('--inference_mode', default='mean_sum', type=str, help='Method used for attention map generation')
    parser.add_argument('--sample_during_training', default=True, type=bool, help='Sample images during training?')
    parser.add_argument('--sample_every_n_epoch', default=5, type=int, help='After how many epochs should we sample')
    
    # Train or test?
    parser.add_argument('--eval', default=False, type=bool, help='Train or only test the model')
    parser.add_argument('--model_version', default=None, type=int, help='Which version of the model to continue training')
    
    args = parser.parse_args()

    exp_vae(args)