import torch
from torch import nn
from torch.nn import Sequential

#model definition
class Unet1D(nn.Module):
    def __init__(self):
        super(Unet1D, self).__init__()
        
        ch = 32
        self.maxpool = nn.MaxPool2d((1,2))
        self.unpool = nn.Upsample(scale_factor=(1,2))
        self.startLayer =  nn.Conv2d(1, ch, (1,3), padding=(0,1))
        self.endLayer = nn.Conv2d(ch, 1, (1,1))
        self.tb1 = Sequential(nn.Conv2d(ch, ch, (1,3), padding=(0,1), bias=False), PReLU())
        self.tb2 = Sequential(nn.Conv2d(ch, ch, (1,3), padding=(0,1), bias=False), PReLU())
        self.tb3 = Sequential(nn.Conv2d(ch, ch, (1,3), padding=(0,1), bias=False), PReLU())
        self.tb4 = Sequential(nn.Conv2d(ch, ch, (1,3), padding=(0,1), bias=False), PReLU())
        self.tb5 = Sequential(nn.Conv2d(ch, ch, (1,3), padding=(0,1), bias=False), PReLU())

        self.db1 = Sequential(nn.Conv2d(ch * 2, ch, (1,3), padding=(0,1), bias=False), PReLU())
        self.db2 = Sequential(nn.Conv2d(ch * 2, ch, (1,3), padding=(0,1), bias=False), PReLU())
        self.db3 = Sequential(nn.Conv2d(ch * 2, ch, (1,3), padding=(0,1), bias=False), PReLU())
        self.db4 = Sequential(nn.Conv2d(ch * 2, ch, (1,3), padding=(0,1), bias=False), PReLU())
        self.db5 = Sequential(nn.Conv2d(ch, ch, (1,3), padding=(0,1), bias=False), PReLU())


    def forward(self, x):
        data = self.startLayer(x)

        data1 = self.tb1(data)
        data2 = self.tb2(self.maxpool(data1))
        data3 = self.tb3(self.maxpool(data2))
        data4 = self.tb4(self.maxpool(data3))
        data5 = self.tb5(self.maxpool(data4))

        
        data5 = self.db5(data5)
        data4 = self.db4(torch.cat([data4, nn.Upsample(size=(data4.shape[2], data4.shape[3]))(data5)], dim=1))
        data3 = self.db3(torch.cat([data3, nn.Upsample(size=(data3.shape[2], data3.shape[3]))(data4)], dim=1))
        data2 = self.db2(torch.cat([data2, nn.Upsample(size=(data2.shape[2], data2.shape[3]))(data3)], dim=1))
        data1 = self.db1(torch.cat([data1, nn.Upsample(size=(data1.shape[2], data1.shape[3]))(data2)], dim=1))

        return self.endLayer(data1)

#we use cuda for model
model = torch.load("model_unet1d.pkl").cpu()

import numpy as np
#load train and val data
#input sinograms with noise
noised_sin = torch.from_numpy(np.load("data/noised_sin.npy")).unsqueeze(1)
#filtered sinograms without noise
filtered_sin = torch.from_numpy(np.load("data/clear_sin.npy")).unsqueeze(1)
#groundtruth phantoms
phantoms = torch.from_numpy(np.load("data/phantoms.npy")).unsqueeze(1)


import odl
#define radon scheme
detectors = 183
angles = 128
angles_parallel = np.linspace(0, 180, angles, False)

reco_space = odl.uniform_discr(min_pt=[-20,-20], max_pt=[20,20], shape=[128, 128], dtype='float32')

phantom = odl.phantom.shepp_logan(reco_space, modified=True)

import math
l = 40 * math.sqrt(2)

angle_partition = odl.uniform_partition(-np.pi / 2, np.pi / 2, angles)
detector_partition = odl.uniform_partition(-l / 2, l / 2, detectors)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")

def ramp_op(ray_trafo):
    fourier = odl.trafos.FourierTransform(ray_trafo.range, axes=[1])
    # Create ramp in the detector direction
    ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
    # Create ramp filter via the convolution formula with fourier transforms
    ramp_filter = fourier.inverse * ramp_function * fourier
    return ramp_filter

ramp = ramp_op(ray_trafo)

test_data_idx = 1000

inp = noised_sin[test_data_idx:test_data_idx+1]
f_sin = filtered_sin[test_data_idx]
groundtruth = phantoms[test_data_idx, 0].numpy()

#plot and measure experiments
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 3)
fig.set_figheight(15)
fig.set_figwidth(15)

proposed_rec = ray_trafo.adjoint(model(inp).detach().numpy()[0,0]).data
proposed_rec *= (proposed_rec > 0)
fbp_rec = ray_trafo.adjoint(ramp(inp[0,0])).data
fbp_rec *= (fbp_rec > 0)

proposed_diff = np.abs(proposed_rec - groundtruth)
fbp_diff = np.abs(fbp_rec - groundtruth)

# diff_max = max(np.max(proposed_diff), np.max(fbp_diff))
# proposed_diff /= diff_max
# fbp_diff /= diff_max


#show phantom
im_ph = axs[0,0].imshow(groundtruth, cmap='gray')
axs[0,0].set_title('a) Фантом')

#show fbp reconstruction
axs[0,1].imshow(fbp_rec, cmap='gray')
axs[0,1].set_title('б) FBP')
axs[0,1].axhline(y=64, color='orange', ls='--')

#show reconstruction of proposed models
axs[0,2].imshow(proposed_rec, cmap='gray')
axs[0,2].set_title('в) UNet1D')
axs[0,2].axhline(y=64, color='blue', ls='--')


#show diff slice
# axs[1, 2].plot(groundtruth[64], label='Phantom')
axs[1, 0].plot(proposed_rec[64], '-', label='UNet1D', color='blue')
axs[1, 0].plot(fbp_rec[64], '--', label='FBP', color='orange')
axs[1, 0].set_title('г) Срез реконструкции от FBP и UNet1D')
axs[1, 0].grid()
axs[1, 0].legend()

#diff fbp to groundtruth
axs[1,1].imshow(fbp_diff, vmax=groundtruth.max(), vmin=0, cmap='gray')
axs[1,1].set_title('д) Разница между FBP и фантомом')

#diff proposed to groundtruth
axs[1,2].imshow(proposed_diff, vmax=groundtruth.max(), vmin=0, cmap='gray')
axs[1,2].set_title('е) Разница между UNet1D и фантомом')



fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.53, 0.02, 0.35])
fig.colorbar(im_ph, cax=cbar_ax)

plt.show()