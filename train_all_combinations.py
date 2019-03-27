import click
from train_anomaly_detection import main_func
import numpy as np
import os

# Define base parameters.
dataset_name = 'selfsupervised'
net_name = 'StackConvNet'
xp_path_base = 'log'
data_path = 'data/full'
train_folder = 'train'
val_pos_folder = 'val/wangen_sun_3_pos'
val_neg_folder = 'val/wangen_sun_3_neg'
load_config = None
load_model = None
nu = 0.1
device = 'cuda'
seed = -1
optimizer_name = 'adam'
lr = 0.0001
n_epochs = 150
lr_milestone = (100,)
batch_size = 200
weight_decay = 0.5e-6
ae_optimizer_name = 'adam'
ae_lr = 0.0001
ae_n_epochs = 350
ae_lr_milestone = (250,)
ae_batch_size = 200
ae_weight_decay = 0.5e-6
n_jobs_dataloader = 0
normal_class = 1
batchnorm = False
dropout = False
augment = False

objectives = [
{'objective': 'real-nvp',      'pretrain': True,     'fix_encoder': True},   # 0
{'objective': 'soft-boundary', 'pretrain': True,     'fix_encoder': False},  # 1
{'objective': 'one-class',     'pretrain': True,     'fix_encoder': False},  # 2
{'objective': 'real-nvp',      'pretrain': False,    'fix_encoder': False},  # 3
{'objective': 'real-nvp',      'pretrain': True,     'fix_encoder': False},  # 4
{'objective': 'one-class',     'pretrain': False,    'fix_encoder': False},  # 5
{'objective': 'soft-boundary', 'pretrain': False,    'fix_encoder': False}   # 6
]

modalities = [
{'rgb': True , 'ir': False, 'depth': False, 'depth_3d': True , 'normals': False, 'normal_angle': True },
{'rgb': True , 'ir': False, 'depth': False, 'depth_3d': False, 'normals': False, 'normal_angle': False},
{'rgb': False, 'ir': True , 'depth': False, 'depth_3d': False, 'normals': False, 'normal_angle': False},
{'rgb': False, 'ir': False, 'depth': True , 'depth_3d': False, 'normals': False, 'normal_angle': False},
{'rgb': True , 'ir': False, 'depth': True , 'depth_3d': False, 'normals': False, 'normal_angle': False},
{'rgb': False, 'ir': True , 'depth': True , 'depth_3d': False, 'normals': False, 'normal_angle': False},
{'rgb': True , 'ir': True , 'depth': True , 'depth_3d': False, 'normals': False, 'normal_angle': False},
{'rgb': True , 'ir': True , 'depth': True , 'depth_3d': False, 'normals': True , 'normal_angle': False},
{'rgb': True , 'ir': True , 'depth': False, 'depth_3d': True , 'normals': False, 'normal_angle': True },
{'rgb': True , 'ir': False, 'depth': True , 'depth_3d': False, 'normals': True , 'normal_angle': False},
{'rgb': True , 'ir': False, 'depth': False, 'depth_3d': False, 'normals': True , 'normal_angle': False},
{'rgb': True , 'ir': False, 'depth': False, 'depth_3d': True , 'normals': False, 'normal_angle': False},
{'rgb': True , 'ir': False, 'depth': False, 'depth_3d': False, 'normals': False, 'normal_angle': True },
{'rgb': False, 'ir': False, 'depth': True , 'depth_3d': False, 'normals': True , 'normal_angle': False},
{'rgb': False, 'ir': False, 'depth': False, 'depth_3d': True , 'normals': False, 'normal_angle': True },
{'rgb': True , 'ir': False, 'depth': True , 'depth_3d': False, 'normals': False, 'normal_angle': True },
{'rgb': True , 'ir': False, 'depth': False, 'depth_3d': True , 'normals': True , 'normal_angle': False},
{'rgb': False, 'ir': False, 'depth': True , 'depth_3d': False, 'normals': False, 'normal_angle': True }
]

N_ITER = 10

auc_mat = np.zeros((N_ITER, len(objectives)+1, len(modalities)))    # +1 for Autoencoder

for it in range(N_ITER):
  xp_path = os.path.join(xp_path_base, str(it))
  for i, obj in enumerate(objectives):
    for j, mod in enumerate(modalities):
      train_obj = main_func(dataset_name, net_name, xp_path, data_path, train_folder, 
           val_pos_folder, val_neg_folder, load_config, load_model, obj['objective'], nu, 
           device, seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, 
           weight_decay, obj['pretrain'], ae_optimizer_name, ae_lr, ae_n_epochs, 
           ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class, 
           mod['rgb'], mod['ir'], mod['depth'], mod['depth_3d'], mod['normals'], 
           mod['normal_angle'], batchnorm, dropout, augment, obj['fix_encoder'])
      auc = train_obj.results['test_auc']
      auc_ae = train_obj.results['test_auc_ae']
      auc_mat[it, i,j] = auc
      if auc_ae is not None:
        auc_mat[it, -1,j] = auc_ae

  np.save(os.path.join(xp_path, 'auc.npy'), auc_mat)

np.save(os.path.join(xp_path_base, 'auc.npy'), auc_mat)
print('avg')
print(np.mean(auc_mat, axis=0))
print('std')
print(np.std(auc_mat, axis=0))