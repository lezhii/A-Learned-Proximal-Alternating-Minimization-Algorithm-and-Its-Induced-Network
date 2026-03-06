import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import scipy.io as io
import numpy as np
import os
import platform
import scipy.io as sio
from scipy.io import savemat
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from argparse import ArgumentParser
from utils import *
from data import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Load Data

print('Load Data...')

mask = sio.loadmat('dataset_202/mask/mask_202.mat')
m = np.expand_dims(mask['mask_202'].astype(np.float32), axis=0)

#___________________________________________________________________________________Train

print("...................................")
print("Phase Number is %d" % (start_phase ))
print("...................................\n")
print('Load Data...')


#%% training dataloader
if (platform.system() == 'Windows'):
    rand_loader = DataLoader(dataset, 
                             batch_size=1, num_workers=0,shuffle=False)
else:
    rand_loader = DataLoader(dataset, 
                             batch_size=1, num_workers=0,shuffle=False)#?


#%% testing dataloader
if (platform.system() == 'Windows'):
    randtest_loader = DataLoader(dataset_test, 
                             batch_size=1, num_workers=0,shuffle=False)
else:
    randtest_loader = DataLoader(dataset_test, 
                             batch_size=1, num_workers=0,shuffle=False)#?                             

#%% initialize model

model = LDA(layer_num, start_phase)
model = nn.DataParallel(model)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/LDA_layer_%d_group_%d_ratio_lr_%.4f" % \
    (args.model_dir, layer_num, group_num,  learning_rate)

log_file_name = "./%s/LDA_layer_%d_group_%d_ratio_lr_%.4f.txt" % \
    (args.log_dir, layer_num, group_num,  learning_rate)
    
log_file_name1 = "./%s/LDA_layer_%d_group_%d_ratio_lr_%.4f.txt" % \
    (args.log_dir, layer_num, group_num,  learning_rate)
    
# if start from beginning load pretrained models

# model.load_state_dict(torch.load('%s/net_params_epoch%d_phase%d.pkll_kx3' % \
#                                     (model_dir, 2, 5), 
#                                     map_location=device),strict=False)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

#%% training
for PhaseNo in range(start_phase, end_phase+1, 2):
    # add new phases
    model.module.set_PhaseNo(PhaseNo)
    if PhaseNo == 3:
        end_epoch = 50
    else:
        end_epoch = args.end_epoch
    for epoch_i in range(start_epoch+1, end_epoch+1):
        progress = 0
        for kx1,kx2,t1,t2,k1,k2 in rand_loader:
            
            kspace1 = k1
            kspace1 = kspace1.to(device)
        
            kspace2 = k2
            kspace2 = kspace2.to(device)
            
            kx1 = kx1.to(device)
            kx2 = kx2.to(device)
            
            t1 = t1.to(device)
            t2 = t2.to(device)
            
            k11,k22 = model(PhaseNo,kspace1,kspace2,kx1,kx2)
            
            k11=torch.abs(k11)
            k22=torch.abs(k22)
            k33=torch.abs(k33) 
            
            cost_tr_rec1 = MSE(t1, k11)
            cost_tr_rec2 = MSE(t2, k22)
            
            ssim1= ssimloss(k11,t1)
            ssim2= ssimloss(k22,t2)
            
            loss_all = cost_tr_rec1 + cost_tr_rec2 + 0.1*ssim1 + 0.1*ssim2 
            
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            nrtrain = 500
            
            if progress % 100 == 0:
                output_data = "[Phase %02d] [Epoch %02d/%02d] Total Loss: %.4f" % \
                    (PhaseNo, epoch_i, end_epoch, loss_all.item()) \
                    + "\t progress: %02f" % (progress * batch_size/ nrtrain * 100) + "%\n"
            
                print(output_data)

       
        if epoch_i % 10 == 0:
            torch.save(model.state_dict(), "./%s/net_params_epoch%d_phase%d.pkll_kx3" % \
                      (model_dir, epoch_i, PhaseNo))
                
 #%% testing
print("Load Testing Data")


print(model.load_state_dict(torch.load('%s/net_params_epoch%d_phase%d.pkll_kx3' % \
                                     (model_dir, 30, 15))))
                                     
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print(model.state_dict()[param_tensor])



model.to(device)

TIME_ALL  = []

PSNR1_All = []
SSIM1_All = []
NMSE1_All = []
RMSE1_All = []

psnr_t1_list = []
ssim_t1_list = []

psnr_t2_list = []
ssim_t2_list = []


init_x1 = []
init_x2 = []

progress = 0

with torch.no_grad():
    for kx1,kx2,t1,t2,k1,k2 in randtest_loader:
        # Move data to device
        progress += 1
        kspace1 = k1
       
        kspace1 = kspace1.to(device)
        
        kspace2 = k2
        
        kspace2 = kspace2.to(device)
        
        kx1 = kx1.to(device)
    
        kx2 = kx2.to(device)
        
        t1 = t1.to(device)
        t2 = t2.to(device)
        
        
        k11,k22= model(15,kspace1,kspace2,kx1,kx2)
        
        # Convert complex outputs to magnitude
        k11 = torch.abs(k11)
        k22 = torch.abs(k22)
        
        # Move tensors to CPU and convert to numpy
        t1_np = t1.squeeze().cpu().numpy()
        t2_np = t2.squeeze().cpu().numpy()
    
        k11_np = k11.squeeze().cpu().numpy()
        k22_np = k22.squeeze().cpu().numpy()
        
        # Compute PSNR
        psnr_t1_list.append(psnr(t1_np, k11_np,data_range=1.0))
        psnr_t2_list.append(psnr(t2_np, k22_np,data_range=1.0))
        
        # Compute SSIM
        ssim_t1_list.append(ssim(t1_np, k11_np,data_range=1.0))
        ssim_t2_list.append(ssim(t2_np, k22_np,data_range=1.0))

# Convert lists to numpy arrays for stats
psnr_t1_arr = np.array(psnr_t1_list)
psnr_t2_arr = np.array(psnr_t2_list)

ssim_t1_arr = np.array(ssim_t1_list)
ssim_t2_arr = np.array(ssim_t2_list)

# Print mean and std for each modality
print("T1 PSNR: Mean = {:.4f}, Std = {:.4f}".format(psnr_t1_arr.mean(), psnr_t1_arr.std()))
print("T1 SSIM: Mean = {:.4f}, Std = {:.4f}".format(ssim_t1_arr.mean(), ssim_t1_arr.std()))

print("T2 PSNR: Mean = {:.4f}, Std = {:.4f}".format(psnr_t2_arr.mean(), psnr_t2_arr.std()))
print("T2 SSIM: Mean = {:.4f}, Std = {:.4f}".format(ssim_t2_arr.mean(), ssim_t2_arr.std()))

    
    # saveAsMat(X1_INIT, "T1_final_5010.mat", 'T1_final_5010',  mat_dict=None)
    # saveAsMat(X2_INIT, "T2_final_5010.mat", 'T2_final_5010',  mat_dict=None)