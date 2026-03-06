import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from argparse import ArgumentParser

import numpy as np
import copy
import math
import scipy.io as io
import os
import gc
# from pytorch_msssim import SSIM,ssim,ms_ssim

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnrr
from skimage.metrics import normalized_root_mse as nmsee
import scipy.io as sio

parser = ArgumentParser(description='Learnable Optimization Algorithms (LOA)')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=30, help='epoch number of end training')
parser.add_argument('--start_phase', type=int, default=3, help='phase number of start training')
parser.add_argument('--end_phase', type=int, default=15, help='phase number of end training')
parser.add_argument('--layer_num', type=int, default=15, help='phase number of LDA-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for loading data')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
# parser.add_argument('--init', type=bool, default=True, help='initialization True by default')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='phi_bar_final_convergence', help='log directory')

args = parser.parse_args()

#%% experiement setup
start_epoch = args.start_epoch
end_epoch = args.end_epoch
start_phase = args.start_phase
end_phase = args.end_phase
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
batch_size = args.batch_size
# init = args.init


#%% define LDA net
class LDA(torch.nn.Module):
    def __init__(self, LayerNo, PhaseNo):
        super(LDA, self).__init__()
        
        # soft threshold
        #0.002
        self.soft_thr1 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr2 = nn.Parameter(torch.Tensor([0.01]))
        # self.soft_thr = 0.01
        # sparcity bactracking
        self.gamma = 1.0
        
        # a parameter for backtracking
        self.sigma = 10000.0
        # parameter for activation function
        self.delta = 0.01
        # set phase number
        self.PhaseNo = PhaseNo
        self.init = True
        
        self.alphas1 = nn.Parameter(0.5 * torch.ones(LayerNo))
        self.alphas2 = nn.Parameter(0.5 * torch.ones(LayerNo))
        self.betas1 = nn.Parameter(0.1 * torch.ones(LayerNo))
        self.betas2 = nn.Parameter(0.1 * torch.ones(LayerNo))
        
        # every block shares weights
        self.conv1r = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2r = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3r = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4r = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        
        self.conv1i = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2i = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3i = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4i = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        
        self.conv1r_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2r_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3r_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4r_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        
        self.conv1i_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2i_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3i_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4i_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        
        mask = sio.loadmat('dataset_202/mask/mask_202.mat')
        m = np.expand_dims(mask['mask_202'].astype(np.float32), axis=0)
        
        self.m = torch.tensor(m).cuda()
        
    def set_PhaseNo(self, PhaseNo):
        # used when adding more phases
        self.PhaseNo = PhaseNo
        
    def activation(self, x):
        """ activation function from eq. (33) in paper """
        
        # index for x < -delta and x > delta
        index = torch.sign(F.relu(torch.abs(x)-self.delta))
        output = index * F.relu(x)
        # add parts when -delta <= x <= delta
        output += (1-index) * (1/(4*self.delta) * torch.square(x) + 1/2 * x + self.delta/4)
        return output
    
    def activation_der(self, x):
        """ derivative of activation function from eq. (33) in paper """
        
        # index for x < -delta and x > delta
        index = torch.sign(F.relu(torch.abs(x)-self.delta))
        output = index * torch.sign(F.relu(x))
        # add parts when -delta <= x <= delta
        output += (1-index) * (1/(2 * self.delta) * x + 1/2)
        return output
    
    def grad_r_x1(self, x1, gamma):
        """ implementation of eq. (10) in paper  """
        
        conv1 = torch.complex(self.conv1r, self.conv1i)
        conv2 = torch.complex(self.conv2r, self.conv2i)
        conv3 = torch.complex(self.conv3r, self.conv3i)
        conv4 = torch.complex(self.conv4r, self.conv4i)
        
        x1_input = x1.view(-1, 1, 160, 180)
        
        soft_thr = self.soft_thr1 * gamma
      
        w1= F.conv2d(x1_input, conv1, padding = 1)
        
        w1_real = torch.real(w1)
        w1_imag = torch.imag(w1)
        w1_real = self.activation(w1_real)
        w1_imag = self.activation(w1_imag)
        w1_act = torch.complex(w1_real,w1_imag)
        
        w2= F.conv2d(w1_act, conv2, padding = 1)
       
        w2_real = torch.real(w2)
        w2_imag = torch.imag(w2)
        w2_real = self.activation(w2_real)
        w2_imag = self.activation(w2_imag)
        w2_act = torch.complex(w2_real,w2_imag)
        
        w3= F.conv2d(w2_act, conv3, padding = 1)
        
        w3_real = torch.real(w3)
        w3_imag = torch.imag(w3)
        w3_real = self.activation(w3_real)
        w3_imag = self.activation(w3_imag)
        w3_act = torch.complex(w3_real,w3_imag)
        
        w4= F.conv2d(w3_act, conv4. padding = 1)

        norm_g = torch.linalg.norm(w4, dim = 1,ord=2)
       
        I1 = torch.sign(F.relu(norm_g - soft_thr))
        I1 = I1.unsqueeze(1) 
        #print((I1 == 1).sum())
        I0 = torch.ones_like(I1) - I1

        ww=I1*w4
  
        g_factor =  F.normalize(ww, dim=1) + I0 * w4 / soft_thr
       
        g1 = F.conv_transpose2d(g_factor, conv4, padding = 1) 
        
        w3_real= torch.real(w3)
        w3_imag= torch.imag(w3)
        w3_real= self.activation_der(w3_real)
        w3_imag= self.activation_der(w3_imag)
        w3=torch.complex(w3_real,w3_imag)
        g1 *= w3
        
        g2 = F.conv_transpose2d(g1, conv3, padding = 1) 
        
        w2_real= torch.real(w2)
        w2_imag= torch.imag(w2)
        w2_real= self.activation_der(w2_real)
        w2_imag= self.activation_der(w2_imag)
        w2=torch.complex(w2_real,w2_imag)
        g2 *= w2
        
        g3 = F.conv_transpose2d(g2, conv2, padding = 1) 
        
        w1_real= torch.real(w1)
        w1_imag= torch.imag(w1)
        w1_real= self.activation_der(w1_real)
        w1_imag= self.activation_der(w1_imag)
        w1=torch.complex(w1_real,w1_imag)
        g3 *= w1
       
        g4 = F.conv_transpose2d(g3, conv1, padding = 1) 
     
        return g4
    
    def grad_r_x2(self, x2, gamma):
        """ implementation of eq. (10) in paper  """
        
        conv1 = torch.complex(self.conv1r, self.conv1i)
        conv2 = torch.complex(self.conv2r, self.conv2i)
        conv3 = torch.complex(self.conv3r, self.conv3i)
        conv4 = torch.complex(self.conv4r, self.conv4i)
        
        x1_input = x1.view(-1, 1, 160, 180)
        
        soft_thr = self.soft_thr2 * gamma
      
        w1= F.conv2d(x1_input, conv1, padding = 1)
        
        w1_real = torch.real(w1)
        w1_imag = torch.imag(w1)
        w1_real = self.activation(w1_real)
        w1_imag = self.activation(w1_imag)
        w1_act = torch.complex(w1_real,w1_imag)
        
        w2= F.conv2d(w1_act, conv2, padding = 1)
       
        w2_real = torch.real(w2)
        w2_imag = torch.imag(w2)
        w2_real = self.activation(w2_real)
        w2_imag = self.activation(w2_imag)
        w2_act = torch.complex(w2_real,w2_imag)
        
        w3= F.conv2d(w2_act, conv3, padding = 1)
        
        w3_real = torch.real(w3)
        w3_imag = torch.imag(w3)
        w3_real = self.activation(w3_real)
        w3_imag = self.activation(w3_imag)
        w3_act = torch.complex(w3_real,w3_imag)
        
        w4= F.conv2d(w3_act, conv4. padding = 1)

        norm_g = torch.linalg.norm(w4, dim = 1,ord=2)
       
        I1 = torch.sign(F.relu(norm_g - soft_thr))
        I1 = I1.unsqueeze(1) 
        #print((I1 == 1).sum())
        I0 = torch.ones_like(I1) - I1

        ww=I1*w4
  
        g_factor =  F.normalize(ww, dim=1) + I0 * w4 / soft_thr
       
        g1 = F.conv_transpose2d(g_factor, conv4, padding = 1) 
        
        w3_real= torch.real(w3)
        w3_imag= torch.imag(w3)
        w3_real= self.activation_der(w3_real)
        w3_imag= self.activation_der(w3_imag)
        w3=torch.complex(w3_real,w3_imag)
        g1 *= w3
        
        g2 = F.conv_transpose2d(g1, conv3, padding = 1) 
        
        w2_real= torch.real(w2)
        w2_imag= torch.imag(w2)
        w2_real= self.activation_der(w2_real)
        w2_imag= self.activation_der(w2_imag)
        w2=torch.complex(w2_real,w2_imag)
        g2 *= w2
        
        g3 = F.conv_transpose2d(g2, conv2, padding = 1) 
        
        w1_real= torch.real(w1)
        w1_imag= torch.imag(w1)
        w1_real= self.activation_der(w1_real)
        w1_imag= self.activation_der(w1_imag)
        w1=torch.complex(w1_real,w1_imag)
        g3 *= w1
       
        g4 = F.conv_transpose2d(g3, conv1, padding = 1) 
     
        return g4    
    
    
    def phase(self, input1,input2, input3, phase,kspace1,kspace2):
        

        """
        input1,input2 is the reconstruction output from last phase
        y is Phi True_x, the sampled ground truth
        
        """
        alpha1 = torch.abs(self.alphas1[phase])
        tau1 = torch.abs(self.betas1[phase])
        alpha2 = torch.abs(self.alphas2[phase])
        tau2 = torch.abs(self.betas2[phase])
        
        #x1_update
     
        ATf1 = mriAdjointOp(kspace1, self.m)#FTPT(f1)
        ATAx1 = mriAdjointOp(mriForwardOp(input1, self.m), self.m)# FTPT(PFx1)
        
        z1 = input1 - alpha1 * (ATAx1 - ATf1) 
        g_x1 = self.grad_r_x1(z1, gamma)
        u1 = z1 - tau1 * g_x1
        
        #x2_update
        
        ATAx2 = mriAdjointOp(mriForwardOp(input2, self.m), self.m)# FTPT(PFx1)
        ATf2  = mriAdjointOp(kspace2, self.m)#FTPT(f1)
        
        z2 = input2 - alpha2 * (ATAx2 - ATf2)
        g_x2 = self.grad_r_x2(z2, gamma)
        u2 = z2 - tau2 * g_x2
        
        x1 = u1
        x2 = u2


        return x1,x2
        
    def forward(self, n,k1,k2,kx1,kx2):
        layers1=[]        
        layers2=[]

        kx1=torch.view_as_complex(kx1)
        kx2=torch.view_as_complex(kx2)

        k1=torch.view_as_complex(k1)
        k2=torch.view_as_complex(k2)
        
        layers1.append(kx1)
        layers2.append(kx2)
      
        gamma=1
        

        for phase in range(self.PhaseNo):
            [k11,k22]=self.phase(layers1[-1],layers2[-1],phase,k1,k2)
                
            layers1[-1] = k11
            layers2[-1] = k22
               
        return layers1[-1], layers2[-1]

        
#%% helper functions
def mriForwardOp(img, sampling_mask):
    # centered Fourier transform
    Fu = torch.fft.fftn(img)
    # apply sampling mask
    Fu_shifted = torch.fft.fftshift(Fu)
    kspace = torch.complex(torch.real(Fu_shifted) * sampling_mask, torch.imag(Fu_shifted) * sampling_mask)
    return kspace

def mriAdjointOp(f, sampling_mask):
    # apply mask and perform inverse centered Fourier transform
    
    Finv = torch.fft.ifftn(torch.fft.ifftshift(torch.complex(torch.real(f) * sampling_mask, torch.imag(f) * sampling_mask)))
    return Finv 

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)
    
def psnrr(img1, img2):
    mse = torch.mean((abs(img1) - abs(img2)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#order matters    
def nmse(img_t, img_s ):
    return torch.sum((abs(img_s) - abs(img_t)) ** 2) / torch.sum(abs(img_t)**2)

#RMSE
def rmse(img_t, img_s ):
    return torch.sqrt(torch.sum((abs(img_s) - abs(img_t)) ** 2) / (160*180) )
    
    
def saveAsMat(img, filename, matlab_id, mat_dict=None):
    """ Save mat files with ndim in [2,3,4]
        Args:
            img: image to be saved
            file_path: base directory
            matlab_id: identifer of variable
            mat_dict: additional variables to be saved
    """
    assert img.ndim in [2, 3, 4]

    img_arg = img.copy()
    if img.ndim == 3:
        img_arg = np.transpose(img_arg, (0,1,2))
    elif img.ndim == 4:
        img_arg = np.transpose(img_arg, (2, 3, 0, 1))

    if mat_dict == None:
        mat_dict = {matlab_id: img_arg}
    else:
        mat_dict[matlab_id] = img_arg

    dirname = os.path.dirname(filename) or '.'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    io.savemat(filename, mat_dict)
    
def ssimloss(X, Y):
    assert not torch.is_complex(X)
    assert not torch.is_complex(Y)
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = 1
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)
    uy = F.conv2d(Y, w)
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux ** 2 + uy ** 2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return 1 - S.mean()