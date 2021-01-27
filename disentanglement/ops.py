"""ops.py"""
from __future__ import print_function

import torch
import torch.nn.functional as F


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


def attention_disentanglement(att1, att2):
    """
        Implementation of the equation 5 from author's paper
        ARGS:
            att1: attention map A1
            att2: attention map A2
        Returns:
            Float value representing the loss
    """
    # TODO assert expected shapes and types
    numer = torch.min(att1,att2).sum()
    denomi = torch.add(att1,att2).sum()

    return 2 * (numer / denomi)


def get_cam(image, gcam):
    """ Combines input img and attention map for final colormap """
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    h, w, d = image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=np.float) + \
        np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    return gcam

########### About GradCam used to compute the attention loss component of the AD-FactorVAE

from collections import OrderedDict

import cv2
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import os


class PropBase(object):

    def __init__(self, model, target_layer, cuda=True):
        self.model = model
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    # set the target class as one others as zero. use this vector for back prop added by Lezi
    def encode_one_hot_batch(self, z, mu, logvar, mu_avg, logvar_avg):
        one_hot_batch = torch.FloatTensor(z.size()).zero_()
        return mu

    def forward(self, x):
        self.preds = self.model(x)
        self.image_size = x.size(-1)
        recon_batch, self.mu, self.logvar = self.model(x)
        return recon_batch, self.mu, self.logvar

    # back prop the one_hot signal
    def backward(self, mu, logvar, mu_avg, logvar_avg):
        self.model.zero_grad()
        z = self.model.reparameterize_eval(mu, logvar).cuda()
        one_hot = self.encode_one_hot_batch(z, mu, logvar, mu_avg, logvar_avg)

        if self.cuda:
            one_hot = one_hot.cuda()
        flag = 2
        if flag == 1:
            self.score_fc = torch.sum(F.relu(one_hot.cuda() * mu))
        else:
            self.score_fc = torch.sum(one_hot.cuda())
        self.score_fc.backward(gradient=one_hot, retain_graph=True)

    def get_conv_outputs(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))


class GradCamDissen(PropBase):
    """
        Extend the baseline class and implements the FactorVAE pipeline
            with equation (5) from the paper
    """
    def __init__(self, VAE, discrim, target_layer, cuda=True):
        super(GradCamDissen, self).__init__(VAE,target_layer)
        self.D = discrim

    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0].cpu()

        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
            module[1].register_forward_hook(func_f)

    def forward(self, x):
        self.image_size = x.size(-1)
        recon_batch, self.mu, self.logvar, self.z = self.model(x)
        return recon_batch, self.mu, self.logvar, self.z

    def backward(self, mu, logvar, mu_avg, logvar_avg, flag=2):
        self.model.zero_grad()
        one_hot = mu  # self.encode_one_hot_batch(self.z, mu, logvar, mu_avg, logvar_avg)

        if flag == 1:
            self.score_fc = torch.sum(F.relu(one_hot.cuda() * mu))
        else:
            self.score_fc = torch.sum(one_hot.cuda())

        self.score_fc.backward(gradient=one_hot, retain_graph=True)

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def compute_gradient_weights(self):
        self.grads = self.normalize(self.grads)
        self.map_size = self.grads.size()[2:]
        self.weights = nn.AvgPool2d(self.map_size)(self.grads)

    def generate(self):
        # get gradient
        self.grads = self.get_conv_outputs(
            self.outputs_backward, self.target_layer)

        # compute weithts based on the gradient
        self.compute_gradient_weights()

        # get activation
        self.activiation = self.get_conv_outputs(
            self.outputs_forward, self.target_layer)

        self.weights.volatile = False
        self.activiation = self.activiation[None, :, :, :, :]
        self.weights = self.weights[:, None, :, :, :]

        gcam = F.conv3d(self.activiation, (self.weights.cuda()), padding=0, groups=len(self.weights))
        gcam = gcam.squeeze(dim=0)
        gcam = F.upsample(gcam, (self.image_size, self.image_size), mode="bilinear")
        gcam = torch.abs(gcam)

        return gcam
