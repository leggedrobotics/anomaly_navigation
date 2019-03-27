import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import cv2

from anomaly_detection.datasets.selfsupervised_images import SelfSupervisedDataset



SQ_SIZE=32



def getPermutedNumpyPatchesFromPytorch(images, x_ind, y_ind):
  patches = []
  for img in images:
    img = img.numpy()
    patch = img[0, :, x_ind[0]:x_ind[1], y_ind[0]:y_ind[1]]
    if len(patch.shape) == 3:
      patch = np.transpose(patch, (1, 2, 0))
    else:
      raise Exception('I don\'t think this should ever happen')
    patches.append(np.ascontiguousarray(patch))

  return patches



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



def savePatches(patches, step):
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_rgb.png'), patches[0]*255)

  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_ir.png'), patches[1]*255)

  patches[2].dtype = np.uint8
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_depth.png'), patches[2])

  patches[3].dtype = np.uint8
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_depth_3d_x.png'), patches[3])

  patches[4].dtype = np.uint8
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_depth_3d_y.png'), patches[4])

  patches[5].dtype = np.uint8
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_depth_3d_z.png'), patches[5])

  patches[6].dtype = np.uint8
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_normals_x.png'), patches[6])
  patches[7].dtype = np.uint8
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_normals_y.png'), patches[7])
  patches[8].dtype = np.uint8
  cv2.imwrite(os.path.join(args.outdir, "{:05d}".format(step)+'_normals_z.png'), patches[8])



def label(args):
  dataset = SelfSupervisedDataset(args.datadir, file_format='csv', subsample=args.subsample, tensor_type='float')  
  loader = DataLoader(dataset, shuffle=False)

  if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
    print('Create directory: ' + args.outdir)

  n_steps = len(loader)

  if args.manually:
    for step, (images, labels) in enumerate(loader):
      print(str(step+1) + '/' + str(n_steps),end='\r')

      label_mask = labels != 0
      img_footholds = images[0].clone()
      img_footholds.masked_fill_(label_mask, 0.0)
      img_footholds = img_footholds[0].permute(1,2,0).numpy()
      label_mask = label_mask.squeeze().cpu().numpy()

      fig = plt.figure(figsize=(10, 7), dpi=100)
      ax_img = plt.axes([0.05, 0.15, 0.8, 0.8])
      img_plt = ax_img.imshow(img_footholds)
      ax_patch = plt.axes([0.875, 0.45, 0.1, 0.1])

      def onclick(event):
        # Crop patch.
        # Event data gives images coordinates, whereas we save in matrix coordinates.
        # Thats why we swap x and y
        y_ind = np.array([int(event.xdata)-SQ_SIZE//2, int(event.xdata)+SQ_SIZE//2])
        x_ind = np.array([int(event.ydata)-SQ_SIZE//2, int(event.ydata)+SQ_SIZE//2])
        makeIndexValid(x_ind, y_ind, label_mask.shape)
        patches = getPermutedNumpyPatchesFromPytorch(images, x_ind, y_ind)
        # Draw patch.
        # ax_patch.imshow(patch_rgb)
        # fig.canvas.draw_idle()
        # Save patches.
        savePatches(patches, step)
        plt.close(fig)

      fig.canvas.mpl_connect('button_release_event', onclick)

      plt.show()
  else:
    for step, (images, labels) in enumerate(loader):
      print(str(step+1) + '/' + str(n_steps),end='\r')

      label_mask = (labels!=0).squeeze().numpy()
      foot_ind = np.where(label_mask)
      # Make sure we have a foothold in the image.
      if foot_ind[0].shape[0] == 0:
        print('Encountered empty foothold mask')
        continue

      samp_ind = int(np.random.rand()*foot_ind[0].shape[0])
      indices = np.array([foot_ind[0][samp_ind], foot_ind[1][samp_ind]])

      x_ind = np.array([int(indices[0]-SQ_SIZE/2), int(indices[0]+SQ_SIZE/2)])
      y_ind = np.array([int(indices[1]-SQ_SIZE/2), int(indices[1]+SQ_SIZE/2)])
      makeIndexValid(x_ind, y_ind, label_mask.shape)

      patches = getPermutedNumpyPatchesFromPytorch(images, x_ind, y_ind)

      # Save everything in the appropriate format.
      savePatches(patches, step)



if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--datadir', required=True, help='Directory for dataset')
  parser.add_argument('--outdir', required=True, help='Output directory for patches')
  parser.add_argument('--subsample', type=int, default=1, help='Only use every nth image of the dataset')
  parser.add_argument('--manually', action='store_true', default=False)

  args = parser.parse_args()

  label(args)