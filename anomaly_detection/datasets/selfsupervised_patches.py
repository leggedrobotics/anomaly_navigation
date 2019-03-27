from torch.utils.data import Subset, Dataset, random_split
from PIL import Image
from anomaly_detection.base.torchvision_dataset import TorchvisionDataset

import os
import torchvision.transforms as transforms
from cv2 import imread, IMREAD_UNCHANGED, IMREAD_GRAYSCALE
import numpy as np



# DEPTH_MEAN=1.8502674
# DEPTH_STD=1.566148
DEPTH_MEAN=0.0
DEPTH_STD=10.0



class DatasetCombiner(Dataset):
  def __init__(self, datasets):
    image_list = [dataset.images for dataset in datasets]
    self.images = np.concatenate(tuple(image_list))
    self.labels = []
    for dataset in datasets:
      self.labels += [dataset.label for val in dataset.images]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, ind):
    return self.images[ind], self.labels[ind], ind



class SelfSupervisedDataset(TorchvisionDataset):

  def __init__(self, root: str, train=None, val_pos=None, val_neg=None, rgb=True, ir=False, depth=False, depth_3d=False, normals=False, normal_angle=False, normal_class = 1):
    super().__init__(root)

    if train is None:
      train = 'train'
      print('Set train folder to: ' + train)
    if val_pos is None:
      val_pos = 'wangen_sun_3_pos'
      print('Set val_pos folder to: ' + val_pos)
    if val_neg is None:
      val_neg = 'wangen_sun_3_neg'
      print('Set val_neg folder to: ' + val_neg)

    self.n_classes = 2
    self.normal_class = tuple([normal_class])
    self.outlier_classes = list(range(0, self.n_classes))
    self.outlier_classes.remove(normal_class)

    # transform = transforms.Compose([transforms.ToTensor()])

    # target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

    if not isinstance(train, list) and not isinstance(train, tuple):
      train = (train,)
    if not isinstance(val_pos, list) and not isinstance(val_pos, tuple):
      val_pos = (val_pos,)
    if not isinstance(val_neg, list) and not isinstance(val_neg, tuple):
      val_neg = (val_neg,)

    positive_data_train = MySelfSupervised([os.path.join(root, folder) for folder in train], rgb=rgb, ir=ir, depth=depth, depth_3d=depth_3d, normals=normals, normal_angle=normal_angle, label=1)
    positive_data_val = MySelfSupervised([os.path.join(root, folder) for folder in val_pos], rgb=rgb, ir=ir, depth=depth, depth_3d=depth_3d, normals=normals, normal_angle=normal_angle, label=1)
    negative_data_val = MySelfSupervised([os.path.join(root, folder) for folder in val_neg], rgb=rgb, ir=ir, depth=depth, depth_3d=depth_3d, normals=normals, normal_angle=normal_angle, label=0)

    self.train_set = positive_data_train
    # self.train_set = DatasetCombiner([positive_data_train, unlabelled_data_train])
    # self.test_set = positive_data_val
    self.test_set = DatasetCombiner([negative_data_val, positive_data_val])

    

class MySelfSupervised(Dataset):

  def loadRGBImages(self):
    filenames_rgb = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) 
                                         for f in fn if f.endswith('rgb.png')]
    filenames_rgb.sort()
    print('Found ' + str(len(filenames_rgb)) + ' RGB images in ' + self.root)
    images_rgb = [imread(f) for f in filenames_rgb]
    return images_rgb



  def loadIrImages(self):
    filenames_ir = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) 
                                         for f in fn if f.endswith('ir.png')]
    filenames_ir.sort()
    print('Found ' + str(len(filenames_ir)) + ' IR images in ' + self.root)
    images_ir = [imread(f, IMREAD_GRAYSCALE) for f in filenames_ir]
    return images_ir



  def loadDepthImages(self):
    filenames_depth = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) 
                                         for f in fn if f.endswith('depth.png')]
    filenames_depth.sort()
    print('Found ' + str(len(filenames_depth)) + ' depth images in ' + self.root)
    images_depth = [imread(f, IMREAD_UNCHANGED) for f in filenames_depth]

    # Convert depth to float.
    for image in images_depth:
      if image is not None:
        image.dtype=np.float32

    return images_depth



  def loadDepth3dImages(self):
    filenames_depth_x = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) 
                                                  for f in fn if f.endswith('depth_3d_x.png')]
    filenames_depth_y = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) 
                                                for f in fn if f.endswith('depth_3d_y.png')]
    filenames_depth_z = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) 
                                                for f in fn if f.endswith('depth_3d_z.png')]
    print('Found ' + str(len(filenames_depth_x)) + ' depth_3d images in ' + self.root)
    images_depth_x = [imread(f, IMREAD_UNCHANGED) for f in filenames_depth_x]
    images_depth_y = [imread(f, IMREAD_UNCHANGED) for f in filenames_depth_y]
    images_depth_z = [imread(f, IMREAD_UNCHANGED) for f in filenames_depth_z]

    for image in images_depth_x:
      if image is not None:
        image.dtype=np.float32
    for image in images_depth_y:
      if image is not None:
        image.dtype=np.float32
    for image in images_depth_z:
      if image is not None:
        image.dtype=np.float32

    images_depth_x = np.stack([(np.squeeze(img)-DEPTH_MEAN)/DEPTH_STD for img in images_depth_x if img is not None])
    images_depth_x = np.expand_dims(images_depth_x, axis=1)
    images_depth_y = np.stack([(np.squeeze(img)-DEPTH_MEAN)/DEPTH_STD for img in images_depth_y if img is not None])
    images_depth_y = np.expand_dims(images_depth_y, axis=1)
    images_depth_z = np.stack([(np.squeeze(img)-DEPTH_MEAN)/DEPTH_STD for img in images_depth_z if img is not None])
    images_depth_z = np.expand_dims(images_depth_z, axis=1)

    images_depth_horz = (images_depth_x**2 + images_depth_y**2)**(0.5)

    invalid_mask = (images_depth_horz > 1.0) | (images_depth_z > 1.0) | (images_depth_horz < 0)
    images_depth_horz[invalid_mask] = 0.0
    images_depth_z[invalid_mask] = 0.0
    print('Zeroed out ' + str(np.sum(invalid_mask)) + ' depth 3d values')
    images_depth_3d = np.concatenate([images_depth_horz, images_depth_z], axis=1)

    return images_depth_3d



  def loadNormalsImages(self):
    filenames_normals_x = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) 
                                                for f in fn if f.endswith('normals_x.png')]
    filenames_normals_y = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) 
                                                for f in fn if f.endswith('normals_y.png')]
    filenames_normals_z = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) 
                                                for f in fn if f.endswith('normals_z.png')]
    print('Found ' + str(len(filenames_normals_x)) + ' normals images in ' + self.root)
    images_normals_x = [imread(f, IMREAD_UNCHANGED) for f in filenames_normals_x]
    images_normals_y = [imread(f, IMREAD_UNCHANGED) for f in filenames_normals_y]
    images_normals_z = [imread(f, IMREAD_UNCHANGED) for f in filenames_normals_z]

    for image in images_normals_x:
      if image is not None:
        image.dtype=np.float32
    for image in images_normals_y:
      if image is not None:
        image.dtype=np.float32
    for image in images_normals_z:
      if image is not None:
        image.dtype=np.float32

    images_normals_x = np.stack([np.squeeze(img) for img in images_normals_x if img is not None])
    images_normals_x = np.expand_dims(images_normals_x, axis=1)
    images_normals_y = np.stack([np.squeeze(img) for img in images_normals_y if img is not None])
    images_normals_y = np.expand_dims(images_normals_y, axis=1)
    images_normals_z = np.stack([np.squeeze(img) for img in images_normals_z if img is not None])
    images_normals_z = np.expand_dims(images_normals_z, axis=1)

    images_normals_horz = (images_normals_x**2 + images_normals_y**2)**0.5

    images_normals = np.concatenate([images_normals_horz, images_normals_z], axis=1)    

    return images_normals



  def loadImagesFromFiles(self, label, rgb, ir, depth, depth_3d, normals, normal_angle, indices):
    if rgb:
      images_rgb = self.loadRGBImages()

    if ir:
      images_ir = self.loadIrImages()

    if depth:
      images_depth = self.loadDepthImages()

    if depth_3d:
      images_depth_3d = self.loadDepth3dImages()

    if normals or normal_angle:
      images_normals = self.loadNormalsImages()

    if indices is not None:
      if rgb:
        images_rgb = [images_rgb[ind] for ind in indices]
      if ir:
        images_ir = [images_ir[ind] for ind in indices]
      if depth:
        images_depth = [images_depth[ind] for ind in indices]

    images = []
    if rgb:
      images_rgb = np.stack([img.transpose((2,0,1)).astype(np.float32)/255 for img in images_rgb if img is not None])
      images.append(images_rgb)
    if ir:
      images_ir = np.stack([img.astype(np.float32)/255 for img in images_ir if img is not None])
      images_ir = np.expand_dims(images_ir, axis=1)
      images.append(images_ir)
    if depth:
      images_depth = np.stack([(np.squeeze(img)-DEPTH_MEAN)/DEPTH_STD for img in images_depth if img is not None])
      images_depth = np.expand_dims(images_depth, axis=1)
      images.append(images_depth)
    if depth_3d:
      images.append(images_depth_3d)
    if normals:
      images.append(images_normals)
    if normal_angle:
      normal_angle = np.arctan2(images_normals[:,1:2,:,:], images_normals[:,0:1,:,:])/np.pi
      images.append(normal_angle)

    self.images.append(np.concatenate(images, axis=1))


    self.label = label



  def __init__(self, roots, label, rgb, ir, depth, depth_3d, normals, normal_angle, indices=None):
    super(MySelfSupervised).__init__()

    self.images = []

    for root in roots:
      self.root = root

      self.loadImagesFromFiles(label, rgb, ir, depth, depth_3d, normals, normal_angle, indices)

    self.images = np.concatenate(self.images, axis=0)
    print(str(self.images.shape[0]), 'images total.')



  def __len__(self):
    return len(self.images)



  def __getitem__(self, ind):
    return self.images[ind], self.label, ind

