### Main RBM code taken from: https://github.com/odie2630463/Restricted-Boltzmann-Machines-in-pytorch ###
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
from OneClassMNIST import OneClassMNIST
import cv2
from PIL import Image
import torch.distributions as td


batch_size = 64

# Combines input image and attention map into final colormap
def get_cam(image, gcam):
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

# Define our one class MNIST loaders, one for training, another for testing
train_dataset = OneClassMNIST(1, './Datasets/MNIST_dataset/', transforms.ToTensor(), download=False, train=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size
)

test_dataset = OneClassMNIST(9, './Datasets/MNIST_dataset/', transforms.ToTensor(), download=False, train=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size
)

# Single RBM layer
class RBM(nn.Module):
    def __init__(self,
                 n_vis=784,
                 n_hin=49,
                 k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
    
    def sample_from_p(self,p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
    def v_to_h(self,v):
        v = torch.flatten(v, start_dim=1)
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h
    
    def h_to_v(self,h):
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
    def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_
    
    def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

# A class that stacks RBMs
class StackRBMs(nn.Module):
    def __init__(self, n_hs=[784, 400, 49], layer_idx=1):
        super(StackRBMs, self).__init__()
        self.n_hs = n_hs
        self.layers = nn.Sequential()
        for i, n_h in enumerate(self.n_hs):
            if i == (len(self.n_hs)-1):
                break
            rbm = RBM(n_vis=n_h, n_hin=self.n_hs[i+1], k=1)
            self.layers.add_module(f'rbm{i}', rbm)

        self.layer_idx = layer_idx
        self.curr_layer = 0
    
    def save_layer_grads(self, g_output):
        self.layer_grad_out = g_output
        print(g_output.shape)

    def free_energy(self, v):
        for i, rbm in enumerate(self.layers.children()):
            if i > self.curr_layer:
                break
            if i == self.curr_layer:
                return rbm.free_energy(v)

    def forward(self, v):
        p_h = v
        for i, rbm in enumerate(self.layers.children()):
            if i == self.curr_layer:
                v, v_ = rbm(p_h)
                return v, v_
            p_h, sample_h = rbm.v_to_h(p_h)

    def reconstruct(self, v):
        p_h = v
        for i, rbm in enumerate(self.layers.children()):
            if i > self.curr_layer:
                break
            p_h, sample_h = rbm.v_to_h(p_h)

        p_v = sample_h
        for i, rbm in reversed(list(enumerate(self.layers.children()))):
            if i > self.curr_layer:
                continue
            p_v, sample_v = rbm.h_to_v(p_v)


        return v, p_v, sample_v

    def attmaps(self, v, temp=1.0):
        self.requires_grad_(True)

        v = torch.flatten(v, start_dim=1)
        p_h = v
        for i, layer in enumerate(self.layers.children()):
            if i > self.curr_layer:
                break
            A = F.linear(p_h, layer.W, layer.h_bias)
            if i == self.layer_idx:
                A.register_hook(self.save_layer_grads)
                self.A = A

            p_h = F.sigmoid(A)
        
        sample_h, probs = self.sample_from_p_gumbel(p_h, temp)

        score = torch.sum(sample_h)
        score.backward(retain_graph=True)

        dz_da, A = self.layer_grad_out, self.A

        w = int(np.sqrt(dz_da.shape[1]).item())
        dz_da, A = dz_da.reshape(dz_da.shape[0], 1, w, w), A.reshape(A.shape[0], 1, w, w)
        dz_da = dz_da / (torch.sqrt(torch.mean(torch.square(dz_da))) + 1e-5)
        alpha = F.avg_pool2d(dz_da, kernel_size=dz_da.shape[2:])

        A, alpha = A.unsqueeze(0), alpha.unsqueeze(1)
        M = F.conv3d(A, (alpha), padding=0, groups=len(alpha)).squeeze(0).squeeze(1)
        M = F.interpolate(M.unsqueeze(1), size=(28, 28), mode='bilinear', align_corners=False)
        M = F.sigmoid(M)
        
        v = v.reshape(v.shape[0], 1, 28, 28)
        colormap = self.create_colormap(v, M)

        return M, colormap

    def sample_from_p_gumbel(self, p, temp):
        q_z = td.relaxed_categorical.RelaxedOneHotCategorical(temp, logits=p)  # create a torch distribution
        probs = q_z.probs
        z = q_z.rsample()
        return z, probs

    def create_colormap(self, x, attmaps, unnormalize=True):
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

    def add_layer(self):
        if self.curr_layer+1 >= len(self.layers):
            raise ValueError(f'Cannot add layer, only {len(self.layers)} available')
        self.curr_layer += 1
        for i, layer in enumerate(self.layers.children()):
            if i == self.curr_layer:
                return
            layer.requires_grad_(False)


# Creates a new Stacked RBM of 4 layers trained on OneClassMNIST
def train_srbm():
    srbm = StackRBMs(n_hs=[784, 484, 256, 64, 16])
    train_op = optim.SGD(srbm.parameters(),0.1)
    for epoch in range(80):
        loss_ = []
        for _, (data,target) in enumerate(train_loader):
            data = Variable(data.view(-1,784))
            sample_data = data.bernoulli()
            
            v,v1 = srbm(sample_data)
            loss = srbm.free_energy(v) - srbm.free_energy(v1)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()

        if epoch%20 == 0 and epoch != 0:
            srbm.add_layer()
        
        print(f'Epoch: {epoch}, loss: {np.mean(loss_)}')

    torch.save(srbm.state_dict(), 'srbm.ckpt')
    print('Model saved as srbm.ckpt')
    return srbm

# Loads and returns a stacked RBM model
def load_srbm_model(name='srbm.ckpt'):
    srbm = StackRBMs(n_hs=[784, 484, 256, 64, 16])
    srbm.load_state_dict(torch.load(name))
    srbm.curr_layer = 3
    srbm.layer_idx = 3
    return srbm

# Evaluates a stacked RBM model
def eval_srbm_model(srbm):
    v, _ = next(iter(train_loader))

    v, p_v, sample_v = srbm.reconstruct(v)

    save_image(make_grid(v.reshape(v.shape[0], 1, 28, 28)), f'input.png')
    save_image(make_grid(sample_v.reshape(v.shape[0], 1, 28, 28)), f'reconstruct.png')

    v_outlier, _ = next(iter(test_loader))

    M, colormap = srbm.attmaps(v_outlier)
    save_image(make_grid(M), f'outlier_input.png')
    save_image(make_grid(colormap), f'outlier_attmaps.png')

    for i in range(4):
        srbm.layer_idx = i
        srbm.curr_layer = i
        M, colormap = srbm.attmaps(v_outlier)
        save_image(make_grid(M), f'outlier_input_layerid{i}.png')
        save_image(make_grid(colormap), f'outlier_attmaps_layerid{i}.png')


# Run this line to train a new stacked RBM
srbm = train_srbm()
# Run this line to load a stacked RBM
# srbm = load_srbm_model()
eval_srbm_model(srbm)
