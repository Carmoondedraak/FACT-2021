import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from operator import mul

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

        # We require different network architectures for different image sizes
        if im_size[1:] == (28, 28):
            # This part defines the encoder part of the VAE
            self.enc_main = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # 28 - 14
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 14 - 7
                nn.LeakyReLU(),
                nn.Flatten(), # 7*7*128 = 6272
                nn.Linear(6272, 1024),
                nn.LeakyReLU()
            )
            # This part defines the decoder part of the VAE
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
            )

        elif im_size[1:] == (100, 100):
            # This part defines the encoder part of the VAE
            self.enc_main = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # 100 - 50
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 50 - 25
                nn.LeakyReLU(),
                nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1), # 25 - 12
                nn.LeakyReLU(),
                nn.Flatten(), # 12*12*128 = 18,432
                nn.Linear(18432, 1024),
                nn.LeakyReLU()
            )
            # This part defines the decoder part of the VAE
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
            )

        self.mu = nn.Linear(1024, self.z_dim)
        self.var = nn.Linear(1024, self.z_dim)
        self.norm_mu = None
        self.norm_log_var = None

        # Create forward and backward functions used to save activations and gradients
        def save_layer_activations(self, input, output):
            self.layer_activ_out = output

        def save_layer_grads(self, g_input, g_output):
            # TODO: Should we take all outputs, or just one? If so, which one?
            # i = torch.randint(low=0, high=g_output.shape[0], size=(1,1)).item()
            # self.layer_grad_out = g_output[i]
            self.layer_grad_out = g_output

        # Register these hooks to the correct module in the network (given by layer_idx)
        self.enc_main[layer_idx].register_forward_hook(save_layer_activations)
        self.enc_main[layer_idx].register_forward_hook(save_layer_grads)

    def encode(self, x):
        """
        Uses encoder part of model to get a mu and log_var for our latent variable z
        """
        h = self.enc_main(x)
        mu, log_var = self.mu(h), self.var(h)
        return mu, log_var

    def decode(self, z):
        """
        Simply uses the decoder of the model to sample images given a latent variable z
        """
        z = self.dec_vae(z)
        return z

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
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(mu.shape).to(mu.device)
        z = mu + eps*std

        return z

    def configure_normal(self, mu, log_var):
        self.norm_mu = mu
        self.norm_log_var = log_var

    def forward(self, x):
        """
        Reconstruct a batch of images for training
        """
        mu, log_var = self.encode(x)
        z = self.sample_reparameterize(mu, log_var)
        x_rec = self.decode(z)
        return x_rec, mu, log_var

    def get_layer_data(self):
        """
        Returns the activations and gradients for the layer chosen for attention map generation
        """
        activ, grad = self.enc_main[self.layer_idx].layer_activ_out, self.enc_main[self.layer_idx].layer_grad_out
        return grad, activ
