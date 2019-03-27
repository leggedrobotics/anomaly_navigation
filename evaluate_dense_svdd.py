import click
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2

from argparse import ArgumentParser

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from anomaly_detection.datasets.selfsupervised_images import SelfSupervisedDataset
from anomaly_detection.networks.main import build_network, build_autoencoder
from anomaly_detection.utils.config import Config


MAX = 1.0
MIN = 0.5
INVALID = -123



class Upsampler:
  def __init__(self, shape):
    # self.upsampler = nn.Upsample(shape)
    self.shape = shape
    self.upsampler = nn.Upsample(scale_factor=4)

  def getPadding(self, tensor):
    cur_shape = tensor.shape
    w_dif = self.shape[-1] - tensor.shape[-1]
    pad_left = w_dif//2
    pad_right = w_dif - pad_left
    h_dif = self.shape[-2] - tensor.shape[-2]
    pad_top = h_dif//2
    pad_bot = h_dif - pad_top
    return (pad_left, pad_right, pad_top, pad_bot)

  def __call__(self, tensor):
    # return self.upsampler(tensor.unsqueeze(0).unsqueeze(0)).squeeze()
    n_missing_dim = 0
    while len(tensor.shape) < 4:
      tensor = tensor.unsqueeze(0)
      n_missing_dim = n_missing_dim + 1
    tensor = self.upsampler(tensor)
    pad = self.getPadding(tensor)
    mean_val = tensor.mean()
    tensor = F.pad(tensor, pad, mode='constant', value=mean_val)
    for i in range(0, n_missing_dim):
      tensor = tensor.squeeze(0)
    return tensor, mean_val



def getOverlayImage(image, mask, invalid_mask):
  mask = mask.squeeze().unsqueeze(-1)
  invalid_mask = invalid_mask.squeeze().unsqueeze(-1)

  mask_red = mask&(~invalid_mask)
  mask_green = ~mask&(~invalid_mask)

  image_out = torch.asin(image.clone()-1)/(math.pi/2)+1.0
  image_out.masked_scatter_(mask_red, (image_out*torch.Tensor(np.array([MAX,MIN,MIN]))).masked_select(mask_red))
  image_out.masked_scatter_(mask_green, (image_out*torch.Tensor(np.array([MIN,MAX,MIN]))).masked_select(mask_green))
  return image_out



def computeObjective(outputs, objective, center):
  if objective == 'real-nvp':
    log_det_J = outputs[1]
    outputs = outputs[0]
  if objective in ['real-nvp']:
    # Reshape so that we can compute log_probs.
    outs_reshape = outputs.permute(0,2,3,1).reshape(-1, outputs.shape[1])
    prior = torch.distributions.MultivariateNormal(center.squeeze(), 
                                                   torch.eye(outputs.shape[1], 
                                                   device=outputs.device))
    log_prob_reshape = prior.log_prob(outs_reshape)
    log_prob = log_prob_reshape.reshape(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
    if objective == 'real-nvp':
      print(log_prob.shape)
      print(log_det_J.shape)
      log_prob = log_prob + log_det_J
    return -log_prob
  elif objective in ['one-class', 'soft-boundary']:
    return ((outputs - center)**2).mean(dim=1)
  else:
    raise Exception('Unknown objective function')




def train(cfg, model):
  assert os.path.exists(cfg.settings['data_path']), "Error: datadir (dataset directory) could not be loaded"

  # Dataset stuff.
  dataset = SelfSupervisedDataset(cfg.settings['data_path'], file_format='csv', subsample=cfg.settings['subsample'], tensor_type='float')
  loader = DataLoader(dataset, num_workers=0, batch_size=cfg.settings['batch_size'], shuffle=False)

  # Figure out base latent weight assuming dense input mask.
  in_shape = dataset[0][0][0].shape

  with torch.no_grad():
    # Determine feature center.
    feature_center = cfg.settings['center']
    print('Features center is')
    print(feature_center)
    # Compute helper matrix with feature center in channel dimension.
    feature_center_2d = feature_center.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    # Check for CUDA.
    if torch.cuda.is_available():
      device = 'cuda:0'
      print('Using CUDA')
    else:
      device = 'cpu'
      print('Using CPU')
    model = model.to(device)
    feature_center_2d = feature_center_2d.to(device)

    upsampler = Upsampler((in_shape[-2], in_shape[-1]))

    last_loss_masks = []

    # Get threshold.
    THRES = (cfg.settings['radius']-10, cfg.settings['radius'], cfg.settings['radius']+10)

    ####### Training ####################

    epoch_steps = len(loader)
    for step, ((images_rgb, images_ir, images_depth, images_depth_3d_x, images_depth_3d_y, images_depth_3d_z, images_normals_x, images_normals_y, images_normals_z), labels) in enumerate(loader):
      print(str(step+1) + '/' + str(epoch_steps),end='\r')

      images_rgb = images_rgb.to(device)
      images_ir = images_ir.to(device)
      images_depth = images_depth.to(device)
      images_depth_3d_x = images_depth_3d_x.to(device)
      images_depth_3d_y = images_depth_3d_y.to(device)
      images_depth_3d_z = images_depth_3d_z.to(device)
      images_normals_x = images_normals_x.to(device)
      images_normals_y = images_normals_y.to(device)
      images_normals_z = images_normals_z.to(device)
      labels = labels.to(device)

      # Normalize depth.
      images_depth = images_depth / 10.0
      images_depth_3d_x = images_depth_3d_x / 10.0
      images_depth_3d_y = images_depth_3d_y / 10.0
      images_depth_3d_z = images_depth_3d_z / 10.0

      # Reconstruct the actual inputs.
      images_normals_horz = (images_normals_x**2 + images_normals_y**2)**0.5
      images_depth_3d_horz = (images_depth_3d_x**2 + images_depth_3d_y**2)**0.5
      images_depth_3d = torch.cat((images_depth_3d_horz, images_depth_3d_z), dim=1)
      images_normals = torch.cat((images_normals_horz, images_normals_z),dim=1)
      images_normal_angle = torch.atan2(images_normals_z, images_normals_horz)/np.pi

      label_mask = (labels != 0)

      images = ()
      if cfg.settings['rgb']:
        images += (images_rgb,)
      if cfg.settings['ir']:
        images += (images_ir,)
      if cfg.settings['depth']:
        images += (images_depth,)
      if cfg.settings['depth_3d']:
        images += (images_depth_3d,)
      if cfg.settings['normals']:
        images += (images_normals,)
      if cfg.settings['normal_angle']:
        images += (images_normal_angle,)
      images_in = torch.cat(images, dim=1)

      features = model(images_in)

      if cfg.settings['objective'] == 'ae':
        loss_unmasked = ((features - images_in)**2).sum(dim=1)
        loss_unmasked = F.avg_pool2d(loss_unmasked, 32, 1)*32
      else:
        loss_unmasked = computeObjective(features, 
                                         cfg.settings['objective'], 
                                         feature_center_2d)

      # Mask image with footholds.
      images_rgb.masked_fill_(label_mask, 0)

      # Get first batch element.
      img = images_in[0,0:3].permute(1,2,0).cpu()
      img_np = img.numpy()

      img_ir = images_depth[0].expand(3, -1, -1).permute(1,2,0).cpu()
      img_ir_np = img_ir.numpy()

      if cfg.settings['objective'] == 'ae':
        img_depth = features[0,0:3].permute(1,2,0).cpu()
      else:
        img_depth = (images_normal_angle[0].expand(3, -1, -1)).permute(1,2,0).cpu()
      img_depth_np = img_depth.numpy()

      loss_img_orig = loss_unmasked[0].squeeze().cpu()
      print(loss_img_orig.shape)
      loss_img_orig, invalid_val = upsampler(loss_img_orig)
      loss_img_np = loss_img_orig.numpy()

      loss_mask = loss_img_orig > THRES[1]

      N_HIST=3

      if cfg.settings['temporal_filter']:
        loss_avg = loss_mask.clone()
        for mask in last_loss_masks:
          loss_avg &= mask
        last_loss_masks.append(loss_mask)
        if len(last_loss_masks) > N_HIST:
          last_loss_masks.pop(0)
        loss_mask = loss_avg

      invalid_mask = loss_img_orig == invalid_val

      img_overlay_np = getOverlayImage(img, loss_mask, invalid_mask).numpy()

      if cfg.settings['save_dir'] is not None:
        # Permute axes and flip RGB.
        print(img_overlay_np.shape)
        img_save = np.flip(img_overlay_np, axis=2)
        file_name = "{:05d}".format(step)+'_inf.png'
        cv2.imwrite(os.path.join(cfg.settings['save_dir'], file_name), img_save*255)

      else:
        # Set up plot with slider.
        fig = plt.figure(figsize=(24, 13), dpi=100)

        ax_loss_img = plt.axes([0.15, 0.1, 0.3, 0.4])
        loss_img_plt = ax_loss_img.imshow(loss_img_orig.numpy())

        ax_img = plt.axes([0.025, 0.55, 0.3, 0.4])
        img_plt = ax_img.imshow(img_np)

        ax_img_ir = plt.axes([0.35, 0.55, 0.3, 0.4])
        img_ir_plt = ax_img_ir.imshow(img_ir_np)

        ax_img_depth = plt.axes([0.675, 0.55, 0.3, 0.4])
        img_depth_plt = ax_img_depth.imshow(img_depth_np)

        ax_img_overlay = plt.axes([0.55, 0.1, 0.3, 0.4])
        
        img_overlay_plt = ax_img_overlay.imshow(img_overlay_np)

        ax_thres = plt.axes([0.25, 0.025, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        s_thres = Slider(ax_thres, 'Threshold', THRES[0], THRES[2], valinit=THRES[1], valstep=(THRES[2]-THRES[0])/1000)

        def update(val):
          threshold_sq = s_thres.val
          loss_img = (loss_img_orig > threshold_sq)
          loss_img_np = loss_img_orig.numpy()
          loss_img_plt.set_data(loss_img_np)
          img_overlay_plt.set_data(getOverlayImage(img, loss_img).numpy())
          fig.canvas.draw_idle()
        s_thres.on_changed(update)

        plt.show()



@click.command()
@click.argument('net_name', type=click.Choice(['StackConvNet', 'fusion']))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary', 'real-nvp', 'ae']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--batch_size', type=int, default=1, help='Batch size for mini-batch training.')
@click.option('--subsample', type=int, default=1, help='Subsample dataset.')
@click.option('--rgb', is_flag=True)
@click.option('--ir', is_flag=True)
@click.option('--depth', is_flag=True)
@click.option('--depth_3d', is_flag=True)
@click.option('--normals', is_flag=True)
@click.option('--normal_angle', is_flag=True)
@click.option('--batchnorm', is_flag=True)
@click.option('--dropout', is_flag=True)
@click.option('--save_dir', type=click.Path(exists=True), default=None)
@click.option('--temporal_filter', default=None)
def main(net_name, data_path, load_model, objective, nu, batch_size, subsample,
         rgb, ir, depth, depth_3d, normals, normal_angle, batchnorm, dropout, 
         save_dir, temporal_filter):
  cfg = Config(locals().copy())

  assert objective in ['one-class','soft-boundary','real-nvp','ae']
  assert net_name in ['StackConvNet', 'fusion']

  if objective == 'ae':
    model = build_autoencoder(net_name, cfg)
  else:
    model = build_network(net_name, cfg)

  test = torch.load(load_model)
  if objective == 'ae':
    out = model.load_state_dict(test['ae_net_dict'], strict=True)
  else:
    out = model.load_state_dict(test['net_dict'], strict=True)
  print('Loading weights output:')
  print(out)
  model = model.cpu()
  # Set radius.
  if objective == 'ae':
    cfg.settings['radius'] = test['thres_ae']
  else:
    cfg.settings['radius'] = test['thres']
  print('Decision radius is',test['thres'])
  # Set center.
  cfg.settings['center'] = torch.Tensor(test['c'])
  model.eval()

  train(cfg, model)



if __name__ == '__main__':
  main()
