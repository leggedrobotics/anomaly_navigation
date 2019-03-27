import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch
import cv2

from anomaly_detection.datasets.selfsupervised_images import SelfSupervisedDataset



SQ_SIZE=32



def correctColor(image):
  image = np.transpose(image, (1, 2, 0))
  image = np.flip(image, axis=2)
  return image



def makeIndexValid(x_ind, y_ind, img_shape):
  x_shape = img_shape[0]
  y_shape = img_shape[1]
  if x_ind[0] < 0:
    x_ind -= x_ind[0]
  if x_ind[1] > x_shape:
    x_ind -= (x_ind[1]-x_shape)

  if y_ind[0] < 0:
    y_ind -= y_ind[0]
  if y_ind[1] > y_shape:
    y_ind -= (y_ind[1]-y_shape)



def saveImages(img_rgb, img_target, step):
  img_rgb = correctColor(img_rgb)
  img_target = correctColor(img_target)
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_rgb.png'), img_rgb*255)
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_target.png'), img_target*255)



def label(args):
  dataset = SelfSupervisedDataset(args.datadir, file_format='csv', subsample=args.subsample, tensor_type='float')  
  loader = DataLoader(dataset, shuffle=False)

  if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
    print('Create directory: ' + args.outdir)

  n_steps = len(loader)

  for step, (images, labels) in enumerate(loader):
    print(str(step+1) + '/' + str(n_steps),end='\r')

    img_rgb = images[0].squeeze().numpy()
    img_target = np.zeros(img_rgb.shape)

    label_mask = (labels!=0).squeeze().numpy()
    foot_ind = np.where(label_mask)
    # Make sure we have a foothold in the image.
    if foot_ind[0].shape[0] == 0:
      print('Encountered empty foothold mask')

    for i in range(foot_ind[0].shape[0]):
      indices = np.array([foot_ind[0][i], foot_ind[1][i]])
      x_ind = np.array([int(indices[0]-SQ_SIZE/2), int(indices[0]+SQ_SIZE/2)])
      y_ind = np.array([int(indices[1]-SQ_SIZE/2), int(indices[1]+SQ_SIZE/2)])
      makeIndexValid(x_ind, y_ind, np.squeeze(img_rgb.shape[1:]))
      patch = img_rgb[:, x_ind[0]:x_ind[1], y_ind[0]:y_ind[1]]
      img_target[:, x_ind[0]:x_ind[1], y_ind[0]:y_ind[1]] = patch

    img_rgb[:, label_mask] = 0.0
    # Erode.
    label_mask = cv2.erode(label_mask.astype(np.uint8)*255, np.ones([5,5], np.uint8),iterations=1).astype(np.bool)
    img_rgb[:, label_mask] = 1.0
    # Save everything in the appropriate format.
    saveImages(img_rgb, img_target, step)



if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--datadir', required=True, help='Directory for dataset')
  parser.add_argument('--outdir', required=True, help='Output directory for patches')
  parser.add_argument('--subsample', type=int, default=1, help='Only use every nth image of the dataset')

  args = parser.parse_args()

  label(args)
