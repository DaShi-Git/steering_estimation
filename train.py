
from dataloader import DepthDatasetLoader
from torch.utils.data import Dataset, DataLoader
import torch
import logging
import sys
import os
import json
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from skimage import io
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
#from utilis import *
from net_model import Net



torch.cuda.empty_cache()

dd_object = DepthDatasetLoader(dataset_directory = "data0000/")

#dd_object[10]
print(len(dd_object))

batch_size = 10
n_val = 100
n_train = 400
# train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
train_set, val_set = random_split(dd_object, [n_train, n_val],generator=torch.Generator().manual_seed(0))
# 3. Create data loaders
loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
print(train_loader)



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Declare Siamese Network
net = Net()
net.to(device=device)

# Decalre Loss Function
#criterion = ContrastiveLoss()
criterion = nn.MSELoss()
# Declare Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=4e-04, weight_decay=0.0005)
#criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.CosineEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')
intrinsics_file = '/home/shida/optical_flow_estimation/3dmotion/camera_intrinsic.json'
with open(intrinsics_file) as f:
    K = json.load(f)
K = np.array(K)
#net.float()
losses=[]
counter=[]
correctnodes=[]
iteration_number = 0
epochs = 1
for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        # with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            
            rgb = batch['rgb']
            #print(rgb1)

            
            rgb = np.transpose(rgb,(0,3,1,2))
            rgb = np.array(rgb)
            rgb = torch.from_numpy(rgb)
            #print(flowlist.shape)
            out = net(rgb.float())
            #print(out)
#             #print(out)
#             #seg0islable = seg0islable*255
#             #print(seg0islable[5438])
            ground=batch['steer']
            loss = criterion(out, ground)
            #loss.float()
            #print(loss)
#             #print(loss)
            loss.backward()
            optimizer.step()    
            print("Epoch {} Counter {} Current loss {}\n".format(epoch,iteration_number,loss.item()))
            iteration_number += 1
            counter.append(iteration_number)
            losses.append(loss.item())
            
            
            
            
            
plt.scatter(counter, losses, alpha=0.6)  
plt.ylabel('Loss',fontsize=15,color='b')
plt.show()

torch.save(net.state_dict(), str('checkpoint_epoch{}.pth'.format(epoch + 1)))
            
                    
            
            
            