import torch
import torch.nn as nn
import numpy as np
from pl_bolts.models.autoencoders.components import (resnet18_encoder, resnet18_decoder)

# Reshape layer, used for going from flat layer to 2D shape in decoder
class Reshape(nn.Module):
    def __init__(self, size):
        super(Reshape, self).__init__()
        
        self.output_size = size

    def forward(self, x):
        x = torch.reshape(x, shape=(-1,*self.output_size))
        return x

class VAE(nn.Module):
    def __init__(self, layer_idx, z_dim, im_size):
        super(VAE, self).__init__()

        self.z_dim, self.layer_idx = z_dim, layer_idx

        # We require different network architectures for different image sizes of the datasets (MNIST, UCSD, MVTEC)

        # Encoder/decoder for MNIST
        if im_size[1:] == (28, 28):
            # This part defines the encoder part of the VAE on images of (1*28*28)
            self.enc_main = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # 28 - 14
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 14 - 7
                nn.LeakyReLU(),
                nn.Flatten(), # 7*7*128 = 6272
                nn.Linear(6272, 1024),
                nn.LeakyReLU()
            )
            # This part defines the decoder part of the VAE on images of (1*28*28)
            self.dec_vae = nn.Sequential(
                nn.Linear(self.z_dim, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 6272),
                nn.LeakyReLU(),
                Reshape((128, 7, 7)),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                # nn.Sigmoid()
            )
            
            # ResNet encoder is saved in self.encoder, but only used for MVTEC dataset. Setting this to None
            # allows us to check which encoder/decoder to use as they differ slightly in input and output
            self.encoder = None

            encoder_output_dim = 1024

        # Encoder/decoder for UCSD
        elif im_size[1:] == (100, 100):
            # This part defines the encoder part of the VAE on images of (1*100*100)
            self.enc_main = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # (100 * 100) - (50 * 50)
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (50 * 50) - (25 * 25)
                nn.LeakyReLU(),
                nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1), # (25 * 25) - (12 * 12)
                nn.LeakyReLU(),
                nn.Flatten(), # 12*12*128 = 18,432
                nn.Linear(18432, 1024),
                nn.LeakyReLU()
            )
            # This part defines the decoder part of the VAE on images of (1*100*100)
            self.dec_vae = nn.Sequential(
                nn.Linear(self.z_dim, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 18432),
                nn.LeakyReLU(),
                Reshape((128, 12, 12)),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                # nn.Sigmoid()
            )
            
            # ResNet encoder is saved in self.encoder, but only used for MVTEC dataset. Setting this to None
            # allows us to check which encoder/decoder to use as they differ slightly in input and output
            self.encoder = None

            encoder_output_dim = 1024
            
        # Encoder/decoder for MVTEC
        elif im_size == (3, 256, 256):
            self.encoder = resnet18_encoder(first_conv=True, maxpool1=True)
            self.decoder = resnet18_decoder(self.z_dim, im_size[1], first_conv=True, maxpool1=True)

            encoder_output_dim = 512

        # Final part of the encoder network, computing mu and log_var
        self.mu = nn.Linear(encoder_output_dim, self.z_dim)
        self.var = nn.Linear(encoder_output_dim, self.z_dim)

        # Norm mu and log_var are only used when we enable "normal_diff" inference
        self.norm_mu = None
        self.norm_log_var = None

        # Create forward and backward functions used to save activations and gradients
        def save_layer_activations(self, input, output):
            input[0].requires_grad_()

            # Save the activation
            self.layer_activ_out = output
            # Create function which will save the gradient during backward pass
            def save_layer_grads(g_output):
                self.layer_grad_out = g_output
            # Register hook to the activation
            self.layer_activ_out.requires_grad_(True)
            self.layer_activ_out.register_hook(save_layer_grads)

        # Register these hooks to the correct module in the network (given by layer_idx)
        if self.encoder is not None:
            for i, (name, module) in enumerate(self.encoder.named_modules()):
                if i == (self.layer_idx + 3):
                    module.register_forward_hook(save_layer_activations)
        else:
            self.enc_main[layer_idx].register_forward_hook(save_layer_activations)

    def encode(self, x):
        """
        Uses encoder part of model to get a mu and log_var for our latent variable z
        """
        if self.encoder is not None:
            h = self.encoder(x)
            mu, log_var = self.mu(h), self.var(h)
        else:
            h = self.enc_main(x)
            mu, log_var = self.mu(h), self.var(h)
        return mu, log_var

    def decode(self, z):
        """
        Simply uses the decoder of the model to sample images given a latent variable z
        """
        if self.encoder is not None:
            x = self.decoder(z)
        else:
            x = self.dec_vae(z)
        return x

    # TODO: Issues on backprop with norm difference reparametriziation
    def norm_diff_reparametrize(self, mu, log_var):
        """
        Alternative reparametrization used only for inferencing attention maps, not used for training
        
        """
        zx = self.sample_reparameterize(self.norm_mu, self.norm_log_var).to(mu.device)
        zy = self.sample_reparameterize(mu, log_var)
        zu = zx + zy

        y_var = torch.exp(log_var)
        norm_var = torch.exp(self.norm_log_var).to(mu.device)

        mu_diff = (self.norm_mu.to(mu.device) - mu)
        var_sum = norm_var + y_var

        first_term = 1/torch.sqrt(2*np.pi*var_sum)
        second_term = torch.exp(-torch.square(zu - mu_diff) / (2*var_sum))

        z = first_term * second_term
        return z


    def sample_reparameterize(self, mu, log_var):
        """
        Reparametrization trick allowing us to sample a latent z, while keeping a deterministic
        path for the gradients to flow to the encoder for trainin
        """
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z, p, q

    def configure_normal(self, mu, log_var):
        """
        Function which allows us to set the norm mu and log_var of the trained class for 
        computing the score using equation 4 of the paper "Towards Visually Explaining Variational Autoencoders"
        """
        self.norm_mu = mu
        self.norm_log_var = log_var

    def forward(self, x):
        """
        Reconstruct a batch of images for training
        """
        mu, log_var = self.encode(x)
        z, p, q = self.sample_reparameterize(mu, log_var)
        x_rec = self.decode(z)
        stats = (p, q, z, mu, log_var)
        return x_rec, stats

    def get_layer_data(self):
        """
        Returns the activations and gradients for the layer chosen for attention map generation
        """
        # For the resnet model, we loop through the model to access the activations and gradients
        if self.encoder is not None:
            for i, (name, module) in enumerate(self.encoder.named_modules()):
                if i == (self.layer_idx + 3):
                    activ, grad = module.layer_activ_out, module.layer_grad_out
                    return activ, grad
        # The other models use nn.Sequential, so we can use the layer_idx directly
        else:
            activ, grad = self.enc_main[self.layer_idx].layer_activ_out, self.enc_main[self.layer_idx].layer_grad_out
        return grad, activ
