import click
from train_anomaly_detection import main_func
from anomaly_detection.datasets.main import load_dataset
import numpy as np
import os

# Define base parameters.
dataset_name = 'selfsupervised'
net_name = 'StackConvNet'
xp_path_base = 'log'
data_path = 'data/incremental'
train_folders = ('train/base', 'train/wangen_sun_1_pos','train/wangen_twilight_2_pos','train/wangen_rain_2_pos')
val_pos_folders = ('val/wangen_sun_3_pos','val/wangen_fire_4_pos','val/wangen_rain_1_pos','val/wangen_wet_1_pos','val/wangen_twilight_1_pos')
val_neg_folders = ('val/wangen_sun_3_neg','val/wangen_fire_4_neg','val/wangen_rain_1_neg','val/wangen_wet_1_neg','val/wangen_twilight_1_neg')
load_config = None
load_model = None
nu = 0.1
device = 'cuda'
seed = -1
optimizer_name = 'adam'
lr = 0.0001
# n_epochs = 150
n_epochs = 10
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

objective = 'real-nvp'
pretrain = True
fix_encoder = True
rgb = True
ir = False
depth = True
depth_3d = False
normals = False
normal_angle = False

class Config:
  def __init__(self):
    self.settings={}

cfg = Config
cfg.settings = {
# 'train_folder': train_folder
# 'val_pos_folder': val_pos_folder
# 'val_neg_folder': val_neg_folder
'rgb': rgb,
'ir': ir,
'depth': depth,
'depth_3d': depth_3d,
'normals': normals,
'normal_angle': normal_angle,
}


N_ITER = 10

auc_mat = np.zeros((N_ITER, len(train_folders), len(val_pos_folders)))
fpr5_mat = np.zeros((N_ITER, len(train_folders), len(val_pos_folders)))

for it in range(N_ITER):
  xp_path = os.path.join(xp_path_base, str(it))
  for i, _ in enumerate(train_folders):
    train_folder = train_folders[:i+1]
    cfg.settings['train_folder'] = train_folder

    train_obj = main_func(dataset_name, net_name, xp_path, data_path, train_folder, 
         val_pos_folders[0], val_neg_folders[0], load_config, load_model, objective, nu, 
         device, seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, 
         weight_decay, pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, 
         ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class, 
         rgb, ir, depth, depth_3d, normals, 
         normal_angle, batchnorm, dropout, augment, fix_encoder)
    for j, (val_pos_folder, val_neg_folder) in enumerate(zip(val_pos_folders, val_neg_folders)):
      cfg.settings['val_pos_folder'] = val_pos_folder
      cfg.settings['val_neg_folder'] = val_neg_folder
      dataset_val = load_dataset(dataset_name, data_path, normal_class, cfg)
      train_obj.test(dataset_val, device, n_jobs_dataloader)

      auc_mat[it, i,j] = train_obj.results['test_auc']
      fpr5_mat[it, i,j] = train_obj.results['test_fpr5']
      # Get ROC curve file name.
      train_name = train_folder[-1].split('/')[-1]
      test_name = val_pos_folder.split('/')[-1]
      file_name = 'roc_curve_' + train_name + '_' + test_name + '.svg'
      train_obj.trainer.roc_plt[0].savefig(os.path.join(xp_path, file_name))

  # print(auc_mat[it])
  np.save(os.path.join(xp_path, 'auc.npy'), auc_mat)
  np.save(os.path.join(xp_path, 'fpr5.npy'), fpr5_mat)

np.save(os.path.join(xp_path_base, 'auc.npy'), auc_mat)
np.save(os.path.join(xp_path_base, 'fpr5.npy'), fpr5_mat)
print('AUC avg')
print(np.mean(auc_mat, axis=0))
print('AUC std')
print(np.std(auc_mat, axis=0))
print('FPR5 avg')
print(np.mean(fpr5_mat, axis=0))
print('FPR5 std')
print(np.std(fpr5_mat, axis=0))
