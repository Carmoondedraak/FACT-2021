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
from pytorch_lightning import metrics

# Set seeds for reproducibility
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

class ExpVAE(pl.LightningModule):
    
    def __init__(self, im_shape, lr=1e-3, inference_mode='mean_sum', layer_idx=2, z_dim=32, auroc=False):
        """
        args:
            lr - Learning rate used for training
            inference_mode - Which kind of inferencing to use to calculate the score to be backpropagated on
                        options are 'mean_sum' and 'normal_diff'
            layer_idx - Index for which layer of the network to use the activations and gradients for for 
                        attention map generation
            z_dim - The latent dimension size to use for encoding/decoding
            auroc - Boolean switch, indicating whether we have access to target masks and if we want to compute
                    the AUROC scores for them. MNIST doesn't have target masks, UCSD and MVTEC do
        """
        super().__init__()

        self.save_hyperparameters()
        
        self.vae = VAE(self.hparams.layer_idx, self.hparams.z_dim, self.hparams.im_shape)

        if self.hparams.auroc:
            self.roc = metrics.ROC(pos_label=1)

    def loss_f(self, x_rec, x, stats):
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
            # L_rec = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
            L_rec = F.binary_cross_entropy(x_rec, x, reduction='sum')
            # L_rec = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # For color, we use mean MSE for reconstruction loss)
        elif n_channels == 3:
            L_rec = F.mse_loss(x_rec, x, reduction='mean')

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
            if self.hparams.auroc:
                x, ground_truth = batch
                ground_truth = ground_truth.to(self.device)

                # Compute attention maps
                _, attmaps, _ = self.forward(x)

                # Update the ROC with new predictions and ground truths
                self.roc.update(attmaps, ground_truth)

    def validation_epoch_end(self, outputs):
        """
        After going through the entire validation set, we compute the final ROC curve accumulated overall
        predictions and target masks, then compute the AUROC
        """
        if self.hparams.auroc:
            fpr, tpr, thresholds = self.roc.compute()
            fpr, idx = torch.sort(fpr, descending=False)
            tpr, thresholds = tpr[idx], thresholds[idx]
            auroc = auc(fpr, tpr)
            self.log('auroc', auroc)

    def test_step(self, batch, batch_idx, loader_idx):
        """
        Defines a single testing iteration with the goal of reconstructing the input
        For MNIST, it measures VAE loss with goal of reconstructing the input and attention maps on outlier class images
        For UCSD and MVTEC, it also saves ground truth masks and input images
        """
        x,  _ = batch

        if loader_idx == 0:
            x_rec, stats = self.vae(x)

            loss = self.loss_f(x_rec, x, stats)

            self.log('test_loss', loss)

            return loss
        elif loader_idx == 1:
            _, ground_truth = batch

            x_rec, M, colormaps = self.forward(x)
            x_rec, M, colormaps = x_rec.detach().cpu(), M.detach().cpu(), colormaps.detach().cpu()

            colormaps = make_grid(colormaps)
            save_image(colormaps.float(), f'{self.trainer.logger.log_dir}/batch{batch_idx}-attmaps.png')
            
            # For MNIST, there are no ground truth masks, so we don't compute them
            if self.hparams.auroc:
                x, ground_truth = batch
                ground_truth = ground_truth.to(self.device)

                # Compute attention maps
                _, attmaps, _ = self.forward(x)

                # Update the ROC with new predictions and ground truths
                self.roc.update(attmaps, ground_truth)

                # Save ground truth and input images
                ground_truth = make_grid(ground_truth)
                save_image(ground_truth.float(), f'{self.trainer.logger.log_dir}/batch{batch_idx}-targets.png')
                input_imgs = make_grid(x)
                save_image(input_imgs.float(), f'{self.trainer.logger.log_dir}/batch{batch_idx}-input.png')

    def test_epoch_end(self, outputs):
        """
        After going through the entire test set, we compute the final ROC curve accumulated overall
        predictions and target masks, then compute the AUROC
        """
        if self.hparams.auroc:
            fpr, tpr, thresholds = self.roc.compute()
            fpr, idx = torch.sort(fpr, descending=False)
            tpr, thresholds = tpr[idx], thresholds[idx]
            auroc = auc(fpr, tpr)
            self.log('auroc', auroc)

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
            # Push images through the network to get reconstruction, and mu for computing the score to backprop on
            x_rec, stats = self.vae(x)
            p, q, z, mu, log_var = stats
            n_channels = x.shape[1]
            
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
        dz_da = dz_da / (torch.sqrt(torch.mean(torch.square(dz_da))) + 1e-5)
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

    def create_colormap(self, x, attmaps, unnormalize=True):
        """
        Creates and returns a colormap from the attention map and original input image
            x - original input images
            attmaps - attention maps from the model inferred from the input images
        """
        if unnormalize:
            x = self.trainer.datamodule.unnormalize_batch(x)
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
        self.roc.update(attmaps, ground_truth)
        # fpr, tpr, thresholds = roc(attmaps, ground_truth, pos_label=1)
        # Get AUC ROC (AUROC) for this ROC curve and log its value
        # auroc = auc(fpr, tpr)
        # self.log('auroc', auroc, prog_bar=True)

        # Get the best binary localization image by picking threshold with best IOU
        # bloc_img, iou_, threshold = self.bloc_from_iou(attmaps, ground_truth, thresholds)

        # Log iou, theshold
        # self.log('best_iou', iou_)
        # self.log('threshold', threshold)

        # return bloc_img

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
            save_image(colormaps_grid, f'{trainer.logger.log_dir}/{self.epoch}-attmaps.png')
            trainer.logger.experiment.add_image('attmaps', colormaps_grid.numpy(), self.epoch)

            # For UCSD and MVTEC, show target masks
            if 'ucsd' in dataset_name or 'mvtec' in dataset_name:
                targets_grid = make_grid(targets)
                save_image(targets_grid.float(), f'{trainer.logger.log_dir}/{self.epoch}-targets.png')
                trainer.logger.experiment.add_image('targets', targets_grid.numpy(), self.epoch)

            # Save binary localization maps for UCSD and MVTEC datasets
            # if 'ucsd' in dataset_name or 'mvtec' in dataset_name:
            #     blocmaps = pl_module.binary_loc_evaluation((imgs, targets))
            #     blocmaps = make_grid(blocmaps.detach().cpu())
            #     save_image(blocmaps.float(), f'{trainer.logger.log_dir}/{self.epoch}-blocmap.png')
            #     trainer.logger.experiment.add_image('blocmaps', blocmaps.numpy(), self.epoch)

        # Save image reconstruction, which is the first dataloader
        elif output_type == 'rec':
            imgs, _ = next(iter(trainer.val_dataloaders[loader_idx]))
            img_rec, _, _ = pl_module.forward(imgs)
            img_rec = img_rec.detach().cpu()
            img_rec = trainer.datamodule.unnormalize_batch(img_rec)
            img_rec = make_grid(img_rec)
            save_image(img_rec, f'{trainer.logger.log_dir}/{self.epoch}-rec.png')
            trainer.logger.experiment.add_image('rec_images', img_rec.numpy(), self.epoch)
        
        # Also save original input image if enabled
        if include_input:
            img_unnormalized = trainer.datamodule.unnormalize_batch(imgs)
            save_image(img_unnormalized, f'{trainer.logger.log_dir}/{self.epoch}-input.png')

# Main function which initializes the datasets, models, and preps them for training or evaluation
def exp_vae(args):
    # Set working directory to project directory
    set_work_directory()

    # First pick the correct dataset
    if args.dataset.lower() == 'mnist':
        # MNIST can have a train digit and a test/eval digit
        log_dir = 'mnist_logs'
        dm = OneClassMNISTDataModule(root='./Datasets/MNIST_dataset', batch_size=args.batch_size, num_workers=args.num_workers,
                                    train_digit=args.train_digit, test_digit=args.test_digit)

        # MNIST has no masks, so we only qualitatively test it (no AUROC)
        quantitative_eval = False
    elif args.dataset.lower() == 'ucsd':
        # UCSD has pedestrians in the training images, and other vehicles in the test images
        log_dir = 'ucsd_logs'
        dm = UCSDDataModule(root='./Datasets/UCSD_dataset', batch_size=args.batch_size, num_workers=args.num_workers)

        # UCSD has target masks, so we also want to quantitatively evaluate the binary localization maps
        quantitative_eval = True
    elif args.dataset.lower() == 'mvtec':
        log_dir = 'mvtec_logs'
        dm = MVTECDataModule(root='./Datasets/MVTEC_dataset', batch_size=args.batch_size, num_workers=args.num_workers,
                            class_name=args.mvtec_object)
        
        # UCSD has target masks, so we also want to quantitatively evaluate the binary localization maps
        quantitative_eval = True
    elif args.dataset.lower() == 'dsprites':
        log_dir = 'dsprites_logs'
        raise NotImplementedError

    # Create PyTorch Lightning callback for sampling, if enabled
    callbacks = []
    if args.sample_during_training:
        att_map_cb = SampleAttentionCallback(batch_size=args.batch_size, every_n_epoch=args.sample_every_n_epoch)
        callbacks.append(att_map_cb)

    # Create checkpoint for saving the model, based on the validation loss
    # monitor = 'val_loss' if args.dataset.lower() == 'mnist'else 'val_loss/dataloader_idx_0'
    monitor = 'val_loss'
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
            model = ExpVAE(im_size, layer_idx=args.layer_idx, auroc=quantitative_eval)

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
    parser.add_argument('--test_digit', default='7', type=int, help='Digit to be evaluated on (only for MNIST)')
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