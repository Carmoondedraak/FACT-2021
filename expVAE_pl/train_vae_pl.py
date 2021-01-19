import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from PIL import Image
from utils import *
from MNIST_Dataset import OneClassMNISTDataModule
from UCSD_Dataset import UCSDDataModule
from MVTEC_Dataset import MVTECDataModule
from vae_model import VAE
from torchvision.utils import make_grid, save_image
from pytorch_lightning.metrics.functional.classification import roc, auc, iou

# Set seeds for reproducibility
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

    def loss_f(self, recon_x, x, stats):
        """
        Function which calculates the VAE loss using BCE and KL divergence
            x - that original "target" images
            x_rec - reconstructed images from the model
            stats - contains distribution p and q, the sampled latent z, and the output
            of the encoder, mu and _logvar
        """

        # Unpack and define some variables for calculating the loss
        p, q, z, mu, log_var = stats
        n_channels = x.shape[1]
        # For grayscale, we use summed BCE for reconstruction loss
        if n_channels == 1:
            L_rec = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
        # For color, we use mean MSE for reconstruction loss)
        elif n_channels == 3:
            L_rec = F.mse_loss(recon_x, x, reduction='mean')

        # Compute KL divergence between encoder and unit Gaussian
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl = log_qz - log_pz
        kl = kl.mean()
        L_reg = kl*0.1

        # Compute final loss
        loss = L_rec + L_reg
        return loss

    def training_step(self, batch, batch_idx):
        """
        Defines a single training iteration with the goal of reconstructing the input
        """
        x,  _ = batch

        x_rec, stats = self.vae(x)

        loss = self.loss_f(x_rec, x, stats)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, loader_idx):
        """
        Defines a single validation iteration with the goal of reconstructing the input
        """
        x,  _ = batch
        
        # First dataloader contains images of the class we're training on, so we compute
        # regular VAE loss
        if loader_idx == 0:
            x_rec, stats = self.vae(x)

            loss = self.loss_f(x_rec, x, stats)

            self.log('val_loss', loss)

            return loss
        elif loader_idx == 1:
            _, ground_truth = batch
            
            # For MNIST, there are no ground truth masks, so we don't compute them
            if x.shape[2:] == ground_truth.shape[2:]:
                self.binary_loc_evaluation(batch)


    def test_step(self, batch, batch_idx, loader_idx):
        """
        Defines a single testing iteration with the goal of reconstructing the input
        """
        x,  _ = batch

        if loader_idx == 0:
            x_rec, stats = self.vae(x)

            loss = self.loss_f(x_rec, x, stats)

            self.log('test_loss', loss)

            return loss
        elif loader_idx == 1:
            _, ground_truth = batch

            # For MNIST, there are no ground truth masks, so we don't compute them
            if x.shape[2:] == ground_truth.shape[2:]:
                self.binary_loc_evaluation(batch)

    def forward(self, x):
        """
        Forward function which reconstructs input, and also returns attention, color and binary localization maps
        Note that this function is not used during training, only for evaluation
        """

        # Set model to eval, and zero out gradients
        self.eval()
        self.zero_grad()

        # Make sure gradients are enabled (PyTorch Lightning disables gradients for validation loop, which calls this function)
        with torch.set_grad_enabled(True):
            x = x.to(self.device)
            # x.requires_grad_(True)
            # Push images through the network to get reconstruction, and mu for computing the score to backprop on
            x_rec, stats = self.vae(x)
            p, q, z, mu, log_var = stats
            n_channels = x.shape[1]
            if n_channels == 1:
                x_rec = torch.sigmoid(x_rec)
            elif n_channels == 3:
                x_rec = torch.tanh(x_rec)
            
            # For mean sum inference, we simply sum the mu vector to compute the score
            if self.hparams.inference_mode == 'mean_sum':
                score = torch.sum(mu)
            elif self.hparams.inference_mode == 'normal_diff':
                z = self.vae.norm_diff_reparametrize(mu, log_var)
                score = torch.sum(z)

            # Make sure the score 
            if score.requires_grad == False:
                score.requires_grad_()
            score.backward(retain_graph=True)

            # Retrieve the activations and gradients from the specific layer
            dz_da, A = self.vae.get_layer_data()
        
        # We can now compute the attention maps M and create the color maps
        dz_da = dz_da / (torch.sqrt(torch.mean(torch.square(dz_da))))
        alpha = F.avg_pool2d(dz_da, kernel_size=dz_da.shape[2:])
        M = torch.sum(alpha*A, dim=1)
        M = F.interpolate(M.unsqueeze(0), size=self.hparams.im_shape[1:], mode='bilinear', align_corners=False).permute(1,0,2,3)
        M = torch.abs(M)
        colormaps = self.create_colormap(x, M)

        # Zero out the gradients again, and put model back into train mode
        self.zero_grad()
        self.train()

        return x_rec, M, colormaps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def set_normal_stats(self, mu, log_var):
        """
        Sets the VAE's inference method use the difference between the distribution of the 
        trained embeddings vs the distribution of the outlier class
        """
        self.vae.configure_normal(mu=mu, log_var=log_var)
        self.hparams.inference_mode = 'normal_diff'

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
            colormaps[i] = torch.from_numpy(gcam).permute(2,0,1)/255

        permute = [2,1,0]
        colormaps = colormaps[:, permute]
        return colormaps

    def binary_loc_evaluation(self, batch):
        x, ground_truth = batch
        ground_truth = ground_truth.to(self.device)

        # Compute attention maps
        _, attmaps, _ = self.forward(x)

        # Compute the ROC (fpr, tpr) and the thresholds
        fpr, tpr, thresholds = roc(attmaps, ground_truth, pos_label=1)
        # Get AUC ROC (AUROC) for this ROC curve and log its value
        auroc = auc(fpr, tpr)
        self.log('auroc', auroc, prog_bar=True)

        # Get the best binary localization image by picking threshold with best IOU
        bloc_img, iou_, threshold = self.bloc_from_iou(attmaps, ground_truth, thresholds)

        # Log iou, theshold
        self.log('best_iou', iou_)
        self.log('threshold', threshold)

        return bloc_img

    def bloc_from_iou(self, M, target, thresholds, max_search=100):
        """
        Pick the best threshold based off of IOU score, and return the binary localization map, 
        including IOU score and selected threshold
            M - Attention map
            target - Target mask
            thresholds - Thresholds to search
            max_search - How many thresholds we want to search at most
        """
        # Pick 100 evenly spaced thresholds to compute IOU for
        step_size = int(len(thresholds)/max_search)
        thresholds = thresholds[::step_size]

        # Set variables for tracking best iou and thershold
        best_iou = 0
        sel_threshold = None

        # Loop through thresholds, keep track of best iou binary localization map and threshold
        for i, t in enumerate(thresholds):
            m = self.gen_bloc_map(M, t)
            iou_score = iou(m, target)
            if iou_score > best_iou:
                best_iou = iou_score
                sel_threshold = t
                best_bloc = m

        return best_bloc, best_iou, sel_threshold

    def gen_bloc_map(self, M, threshold):
        # Generates a binary localizatino map given a threshold
        M = (M > threshold).to(torch.int)
        return M


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

    def generate_samples(self, trainer, pl_module):
        """
        Function invoked by on_train_epoch_end, invoking sample_images to decide which images to save
        """
        # Generate reconstructed images to see if model is training
        self.sample_images(trainer, pl_module, output_type='rec', include_input=True, loader_idx=0)
        # Generate attentionmaps from "other class" images as well
        self.sample_images(trainer, pl_module, output_type='attmaps', loader_idx=1)

    def sample_images(self, trainer, pl_module, output_type='attmaps', include_input=False, loader_idx=0):
        """
        Function which either saves attention maps, reconstructed images and saves original images
            trainer - PyTorch Lightning trainer object, through which we access the necessary dataloaders
            pl_module - PyTorch Lightning module used for inference
            output_type - Either 'attmaps' or 'rec' for generating the attention maps or reconstructed input images
            include_input - Boolean switch, which when True, also saves original input images
            loader_idx - Int, either 0 or 1, which picks which dataloader to use. 0 is the trained class, 1 is the unseen class
        """

        # Get dataset name
        dataset_name = pl_module.train_dataloader().dataset.__class__.__name__.lower()

        # Save attentionmaps of "other class", which is the second dataloader
        if output_type == 'attmaps':
            imgs, targets = next(iter(trainer.val_dataloaders[loader_idx]))
            _, _, colormaps = pl_module.forward(imgs)
            colormaps = colormaps.detach().cpu()
            colormaps_grid = make_grid(colormaps)
            save_image(colormaps_grid.float(), f'{trainer.logger.log_dir}/{self.epoch}-attmaps.png')
            trainer.logger.experiment.add_image('attmaps', colormaps_grid.numpy(), self.epoch)

            # For UCSD and MVTEC, show target masks
            if 'ucsd' in dataset_name or 'mvtec' in dataset_name:
                targets = make_grid(targets)
                save_image(targets.float(), f'{trainer.logger.log_dir}/{self.epoch}-targets.png')
                trainer.logger.experiment.add_image('targets', targets.numpy(), self.epoch)

            # Save binary localization maps for UCSD and MVTEC datasets
            if 'ucsd' in dataset_name or 'mvtec' in dataset_name:
                imgs, targets = next(iter(trainer.val_dataloaders[loader_idx]))
                blocmaps = pl_module.binary_loc_evaluation((imgs, targets))
                blocmaps = make_grid(blocmaps.detach().cpu())
                save_image(blocmaps.float(), f'{trainer.logger.log_dir}/{self.epoch}-blocmap.png')
                trainer.logger.experiment.add_image('blocmaps', blocmaps.numpy(), self.epoch)

        # Save image reconstruction, which is the first dataloader
        elif output_type == 'rec':
            imgs, _ = next(iter(trainer.val_dataloaders[loader_idx]))
            img_rec, _, _ = pl_module.forward(imgs)
            img_rec = img_rec.detach().cpu()
            img_rec = make_grid(img_rec)
            save_image(img_rec.float(), f'{trainer.logger.log_dir}/{self.epoch}-rec.png')
            trainer.logger.experiment.add_image('rec_images', img_rec.numpy(), self.epoch)
        
        # Also save original input image
        if include_input:
            imgs_grid = make_grid(imgs)
            save_image(imgs_grid.float(), f'{trainer.logger.log_dir}/{self.epoch}-input.png')

# Main function which initializes the datasets, models, and preps them for training or evaluation
def exp_vae(args):
    set_work_directory()

    # First pick the correct dataset
    if args.dataset.lower() == 'mnist':
        log_dir = 'mnist_logs'
        dm = OneClassMNISTDataModule(root='./Datasets/MNIST_dataset', batch_size=args.batch_size, num_workers=args.num_workers,
                                    train_digit=args.train_digit, test_digit=args.test_digit)
    elif args.dataset.lower() == 'ucsd':
        log_dir = 'ucsd_logs'
        dm = UCSDDataModule(root='./Datasets/UCSD_dataset', batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset.lower() == 'mvtec':
        log_dir = 'mvtec_logs'
        dm = MVTECDataModule(root='./Datasets/MVTEC_dataset', batch_size=args.batch_size, num_workers=args.num_workers,
                            class_name=args.mvtec_object)
    elif args.dataset.lower() == 'dsprites':
        log_dir = 'dsprites_logs'
        raise NotImplementedError

    # Create PyTorch Lightning trainer with callback for sampling, if enabled
    callbacks = []
    if args.sample_during_training:
        att_map_cb = SampleAttentionCallback(batch_size=args.batch_size, every_n_epoch=args.sample_every_n_epoch)
        callbacks.append(att_map_cb)

    # Create checkpoint for saving the model, based on the validation loss
    monitor = 'val_loss' if args.dataset.lower() == 'mnist'else 'val_loss/dataloader_idx_0'
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor,
        mode='min',
    )

    # Create PyTorch lightning trainer
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        gpus=1 if torch.cuda.is_available() else 0, 
        checkpoint_callback=checkpoint_cb,
        callbacks=callbacks, 
        max_epochs=args.epochs,
        progress_bar_refresh_rate=1 if args.progress_bar else 0
    )

    # Make sure dataset is prepared/downloaded
    dm.prepare_data()
    dm.setup()

    # Either train or test
    if args.eval:
        # In case of evaluation, we load a pretrained model
        model_path, hparams_path = get_ckpt_path(log_dir, args)
        model = ExpVAE.load_from_checkpoint(
            checkpoint_path=model_path,
            hparams_file=hparams_path
            )

        # Choosing which inferencing method to use
        if args.inference_mode == 'normal_diff':
            mu, var = calc_latent_mu_var(model.vae, dm, args.batch_size)
            model.set_normal_stats(mu, var)

        # Test the model
        trainer.test(model, datamodule=dm)
    else:

        # Load pretrained model if one is specified by model_version
        if args.model_version is not None:
            model_path, hparams_path = get_ckpt_path(log_dir, args)

            model = ExpVAE.load_from_checkpoint(
                checkpoint_path=model_path,
                hparams_file=hparams_path
                )

        # Otherwise, we initialize a new model
        else:
            im_size = dm.dims
            model = ExpVAE(im_size, layer_idx=args.layer_idx)

        # Choosing which inferencing method to use
        if args.inference_mode == 'normal_diff':
            mu, var = calc_latent_mu_var(model.vae, dm, args.batch_size)
            model.set_normal_stats(mu, var)

        # Train model
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
    # Dataset specific options
    parser.add_argument('--train_digit', default='1', type=int, help='Digit to be trained on (only for MNIST)')
    parser.add_argument('--test_digit', default='9', type=int, help='Digit to be evaluated on (only for MNIST)')
    parser.add_argument('--mvtec_object', default='bottle', type=str, help='Object to be trained and evaluated for with MVTEC dataset')
    
    # Inference option
    parser.add_argument('--inference_mode', default='mean_sum', type=str, help='Method used for attention map generation')
    parser.add_argument('--sample_during_training', default=True, type=bool, help='Sample images during training?')
    parser.add_argument('--sample_every_n_epoch', default=10, type=int, help='After how many epochs should we sample')
    
    # Train or test?
    parser.add_argument('--eval', default=False, type=bool, help='Train or only test the model')
    parser.add_argument('--model_version', default=None, type=int, help='Which version of the model to continue training')
    
    args = parser.parse_args()

    exp_vae(args)