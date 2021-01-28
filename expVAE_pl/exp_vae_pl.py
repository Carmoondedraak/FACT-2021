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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class ExpVAE(pl.LightningModule):
    
    def __init__(self, im_shape, auroc=None, lr=None, batch_size=None, epochs=None, progress_bar=None,
                early_stopping=None, layer_idx=None, z_dim=None, dataset=None, num_workers=None,
                train_digit=None, test_digit=None, mvtec_object=None, inference_mode=None,
                sample_during_training=None, sample_every_n_epoch=None, eval=None, model_version=None,
                init_seed=None):
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
            Other arguments are listed are not functional, but for logging, as save_hyperparameters() allows 
            us to track all arguments with tensorboard
        """
        super().__init__()

        # Set defaults to None when arguments are irrelevant
        if 'mvtec' not in dataset:
            mvtec_object = None
        elif 'mnist' not in dataset:
            train_digit, test_digit = None, None

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
        im_shape = x.shape[2:]
        
        # For MVTEC images, we use mean MSE for reconstruction loss
        if im_shape == (256, 256):
            L_rec = F.mse_loss(x_rec, x, reduction='mean')
        # For others, we use summed BCE for reconstruction loss)
        else:
            L_rec = F.binary_cross_entropy(x_rec, x, reduction='sum')

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

    def test_epoch_end(self, outputs):
        """
        After going through the entire test set, we compute the final ROC curve accumulated overall
        predictions and target masks, then compute the AUROC
        """
        if self.hparams.auroc:
            # Compute ROC, then compute AUROC and log the value for the whole test set
            fpr, tpr, thresholds = self.roc.compute()
            fpr, idx = torch.sort(fpr, descending=False)
            tpr, thresholds = tpr[idx], thresholds[idx]
            auroc = auc(fpr, tpr)
            self.log('auroc_test', auroc)

            # Divide thresholds from ROC into 100 equally separated thresholds
            step_size = int(len(thresholds)/100)
            thresholds = thresholds[::step_size]

            # Find best best threshold based off of best IOU
            best_iou = 0
            best_threshold = -1
            # For each threshold, compute IOU for whole test set
            for i, threshold in enumerate(thresholds):
                test_dataloader = self.trainer.datamodule.test_dataloader()[1]
                ious = []
                for batch_idx, (x, y) in enumerate(test_dataloader):
                    x, y = x.to(self.device), y.to(self.device)
                    x_rec, M, colormaps = self.forward(x)
                    bloc_map = self.gen_bloc_map(M, threshold)
                    iou_score = iou(bloc_map, y)
                    ious.append(iou_score.detach().cpu().item())

                avg_iou = np.mean(ious)
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    best_threshold = threshold

                self.trainer.logger.experiment.add_scalar('avg_iou', avg_iou, i)
                self.trainer.logger.experiment.add_scalar('threshold', threshold, i)
            
            # Log best iou and threshold
            self.log('best_iou', best_iou)
            self.log('best_threshold', best_threshold)

            # Now, using best threshold, generate the binary localization maps for 
            # all images in the test set and log/save them
            for batch_idx, (x, y) in enumerate(test_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                x_rec, M, colormaps = self.forward(x)
                bloc_map = self.gen_bloc_map(M, best_threshold)

                # Save the binary localization maps
                bloc_map = bloc_map.detach().cpu()
                bloc_map_grid = make_grid(bloc_map).float()
                save_image(bloc_map_grid, f'{self.trainer.logger.log_dir}/batch{batch_idx}-blocmaps.png')
                self.trainer.logger.experiment.add_image('blocmaps', bloc_map_grid.numpy(), batch_idx)

                # Save the input images
                x = x.detach().cpu()
                x = self.trainer.datamodule.unnormalize_batch(x)
                x_grid = make_grid(x).float()
                save_image(x_grid, f'{self.trainer.logger.log_dir}/batch{batch_idx}-input.png')
                self.trainer.logger.experiment.add_image('input', x_grid.numpy(), batch_idx)

                # Save teh target masks
                y = y.detach().cpu()
                y_grid = make_grid(y).float()
                save_image(y_grid, f'{self.trainer.logger.log_dir}/batch{batch_idx}-targets.png')
                self.trainer.logger.experiment.add_image('targets', y_grid.numpy(), batch_idx)
            
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
        A, alpha = A, alpha

        A, alpha = A.unsqueeze(0), alpha.unsqueeze(1)
        M = F.conv3d(A, (alpha), padding=0, groups=len(alpha)).squeeze(0).squeeze(1)
        M = F.interpolate(M.unsqueeze(1), size=self.hparams.im_shape[1:], mode='bilinear', align_corners=False)
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

    def gen_bloc_map(self, M, threshold):
        # Generates a binary localizatino map given a threshold
        M = (M > threshold).to(torch.int)
        return M


class SampleAttentionCallback(pl.Callback):
    """
    This callback is responsible for sampling attention maps, reconstructed images and original images
    during training. During testing, this is simply done in the test loop
    """
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
    # Set seeds for reproducibility
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    init_seed = args.init_seed
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    np.random.seed(init_seed)

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

    callbacks = []

    # Create PyTorch Lightning callback for sampling, if enabled
    if args.sample_during_training:
        att_map_cb = SampleAttentionCallback(batch_size=args.batch_size, every_n_epoch=args.sample_every_n_epoch)
        callbacks.append(att_map_cb)

    # Create PyTorch Lightning callback for early stopping, in enabled
    if args.early_stopping:
        early_stop_monitor = 'val_loss'
        early_stop_mode = 'min'
        early_stop_cb = EarlyStopping(
            monitor=early_stop_monitor,
            patience=50,
            mode=early_stop_mode
        )
        callbacks.append(early_stop_cb)

    # Create checkpoint callback for saving the model, based on the validation loss or AUROC based on dataset
    monitor = 'val_loss' if args.dataset.lower() == 'mnist'else 'auroc'
    mode = 'min' if args.dataset.lower() == 'mnist' else 'max'
    # monitor = 'val_loss'
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
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
            d = vars(args)
            model = ExpVAE(im_size, quantitative_eval, **d)

        # Choosing which inferencing method to use
        if args.inference_mode == 'normal_diff':
            mu, var = calc_latent_mu_var(model.vae, dm, args.batch_size)
            model.set_normal_stats(mu, var)

        # Train model
        trainer.fit(model, dm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Optimizer and Training Hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for training')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to use for training')
    parser.add_argument('--progress_bar', default=True, type=bool, help='Show or hide progress bar during training')
    parser.add_argument('--early_stopping', default=True, type=bool, help='Enable early stopping of training models. \
                            This is based off of minimum validation loss for MNIST, and maximum AUROC for UCSD and MVTec-AD, with patience 10')

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

    parser.add_argument('--init_seed', default=1, type=int, help='What seed to use for numpy and pytorch')
    
    args = parser.parse_args()

    exp_vae(args)
