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

    def sample_from_p_gumbel(self, p, temp):
        q_z = td.relaxed_categorical.RelaxedOneHotCategorical(temp, logits=p)  # create a torch distribution
        probs = q_z.probs
        z = q_z.rsample()
        return z, probs
    
    def attmap(self, v, temp=1.0):
        v = torch.flatten(v, start_dim=1)
        A = F.linear(v,self.W,self.h_bias)

        def save_layer_grads(g_output):
            self.layer_grad_out = g_output
            print(g_output.shape)

        A.register_hook(save_layer_grads)
        
        p_h = F.sigmoid(A)
        sample_h, probs = self.sample_from_p_gumbel(p_h, temp)

        score = torch.sum(sample_h)
        score.backward(retain_graph=True)

        dz_da = self.layer_grad_out

        # # One version where we reshape at the end
        # dz_da = dz_da / (torch.sqrt(torch.mean(torch.square(dz_da))) + 1e-5)
        # dz_da = dz_da.unsqueeze(1)
        # alpha = F.avg_pool1d(dz_da, kernel_size=dz_da.shape[2]).squeeze(1)
        
        # M = alpha*A
        # M = torch.abs(M)
        # # M = F.relu(M)
        # # M = F.sigmoid(M)

        # w = int(np.sqrt(M.shape[1]).item())
        # M = M.reshape(M.shape[0], 1, w, w)
        # M = F.interpolate(M, size=(28, 28), mode='bilinear', align_corners=False)

        # v = v.reshape(v.shape[0], 1, 28, 28)

        # colormap = self.create_colormap(v, M)

        # Another version where we reshape at the start
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



batch_size = 64

train_dataset = OneClassMNIST(1, './Datasets/MNIST_dataset/', transforms.ToTensor(), download=False, train=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size
)

test_dataset = OneClassMNIST(9, './Datasets/MNIST_dataset/', transforms.ToTensor(), download=False, train=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size
)



rbm = RBM(k=1)




train_op = optim.SGD(rbm.parameters(),0.1)



for epoch in range(100):
    loss_ = []
    for _, (data,target) in enumerate(train_loader):
        data = Variable(data.view(-1,784))
        sample_data = data.bernoulli()
        
        v,v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.item())
        train_op.zero_grad()
        loss.backward()
        train_op.step()
    
    print(np.mean(loss_))

torch.save(rbm.state_dict(), 'model_d1_49z.ckpt')

# NOTE: Uncomment line below to load pretrained model (and comment out the training loop above)
# rbm.load_state_dict(torch.load('./rbm_anomaly_maps/model_d1_49z.ckpt'))


v, _ = next(iter(train_loader))
v, v1 = rbm(v)

save_image(make_grid(v), "real.png")
save_image(make_grid(v1), "generated.png")

v_outlier, _ = next(iter(test_loader))

M, colormap = rbm.attmap(v_outlier, 1.0)
M, colormap = M.cpu().detach(), colormap.cpu().detach()
save_image(make_grid(M), "attmap_test.png")
save_image(make_grid(colormap), "colormap_test.png")

for temp in np.arange(0, 1, 0.1):
    M, colormap = rbm.attmap(v_outlier, temp)
    M, colormap = M.cpu().detach(), colormap.cpu().detach()
    save_image(make_grid(M), f"attmap_{str(temp).replace('.','_')}.png")
    save_image(make_grid(colormap), f"colormap_{str(temp).replace('.','_')}.png")

save_image(make_grid(v_outlier), "outlier_in.png")

print(rbm)






