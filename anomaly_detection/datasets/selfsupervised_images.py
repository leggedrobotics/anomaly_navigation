import math
import numpy as np
import random
import os
import cv2

import torch

from PIL import Image

from numpy import genfromtxt, count_nonzero

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, ColorJitter
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import functional



image_transform = ToPILImage()

IMG_EXTENSIONS = ['.bmp', '.png']

def isImage(file):
    return os.path.splitext(file)[1] in IMG_EXTENSIONS



class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()


class FloatToLongLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)*1000000).long().unsqueeze(0)


class ToFloatLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).float()



def extractStamp(file_name):
  split_name = os.path.basename(file_name).split('_')
  return split_name[0]



def splitByFileBaseName(files):
    active_stamp = -1.
    files.sort()
    files_split = {}
    cur_files = []

    longest_file_list = 0

    for file in files:
        # Get stamp.
        stamp = extractStamp(file)
        # Handle new time stamp.
        if stamp != active_stamp:
            if cur_files:
                files_split[active_stamp] = cur_files
            if len(cur_files) > longest_file_list:
                longest_file_list = len(cur_files)
            cur_files = []
            active_stamp = stamp
        # Append current file.
        cur_files.append(file)

    # One final apend after
    if cur_files:
        files_split[active_stamp] = cur_files

    # Remove lists which are missing some image.
    n_removed = 0
    remove_keys = []
    for key in files_split:
        if len(files_split[key]) != longest_file_list:
            n_removed +=1 
            remove_keys.append(key)
    for key in remove_keys:
        files_split.pop(key)
    if n_removed:
        print('Removed ' + str(n_removed) + ' file lists because they miss some image type.')

    return files_split



class SelfSupervisedDataset(Dataset):

  def isLabel(self, file):
    return file.endswith(self.file_format)

  def __init__(self, root, file_format="npy", subsample=1, tensor_type='long'):
    self.root = root
    self.file_format = file_format
    self.tensor_type = tensor_type
    
    print ("Image root is: " + self.root)
    print ("Load files with extension: " + self.file_format)
    if subsample > 1:
      print("Using every ", subsample, "th image")

    filenames_img = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) for f in fn if isImage(f)]
    filenames_img = splitByFileBaseName(filenames_img)

    filenames_label = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.root), followlinks=True) for f in fn if self.isLabel(f)]
    filenames_label = splitByFileBaseName(filenames_label)

    # Make sure all images have labels and vice-versa.
    keys_remove = []
    for key in filenames_img:
        if not key in filenames_label:
            keys_remove.append(key)
    for key in keys_remove:
        filenames_img.pop(key)
    keys_remove.clear()
    for key in filenames_label:
        if not key in filenames_img:
            keys_remove.append(key)
    for key in keys_remove:
        filenames_label.pop(key)

    # Test to make sure everything's super duper.
    for key1, key2 in zip(filenames_img, filenames_label):
        if key1 != key2:
            raise Exception('Time stampes of images and labels are not identical')

    self.filenames_img = list(filenames_img.values())
    self.filenames_label = list(filenames_label.values())

    if subsample > 1:
      self.filenames_img = [val for ind, val in enumerate(self.filenames_img) if ind % subsample == 0] # Subsample.
      self.filenames_label = [val for ind, val in enumerate(self.filenames_label) if ind % subsample == 0] # Subsample.

    print ("Found " + str(len(self.filenames_img)) + " images.")
    print ("Found " + str(len(self.filenames_label)) + " labels.")



  def __getitem__(self, index):
    filenames_image = self.filenames_img[index]
    filename_labels = self.filenames_label[index]

    image_depth = cv2.imread(filenames_image[0], cv2.IMREAD_UNCHANGED)
    image_depth.dtype=np.float32
    # Remove negative depth which comes from projection.
    image_depth[image_depth < 0.0] = 0.0
    # Remove far away depth.
    image_depth[image_depth > 10.0] = 0.0
    depth_mask = image_depth == 0.0
    image_depth = np.transpose(image_depth, (2, 0, 1))


    image_depth_3d_x = cv2.imread(filenames_image[1], cv2.IMREAD_UNCHANGED)
    image_depth_3d_y = cv2.imread(filenames_image[2], cv2.IMREAD_UNCHANGED)
    image_depth_3d_z = cv2.imread(filenames_image[3], cv2.IMREAD_UNCHANGED)
    image_depth_3d = [image_depth_3d_x, image_depth_3d_y, image_depth_3d_z]
    for i, img in enumerate(image_depth_3d):
        img.dtype=np.float32
        img[depth_mask] = 0.0
        image_depth_3d[i] = np.transpose(img, (2, 0, 1))

    image_bgr = cv2.imread(filenames_image[4], cv2.IMREAD_UNCHANGED).astype(np.float32)/255
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.transpose(image_rgb, (2, 0, 1))

    image_ir = cv2.imread(filenames_image[5], cv2.IMREAD_UNCHANGED).astype(np.float32)/255
    image_ir = np.expand_dims(image_ir, 0)

    image_normal_x = cv2.imread(filenames_image[6], cv2.IMREAD_UNCHANGED).astype(np.float32)    # Normals are uint8 scaled
    image_normal_y = cv2.imread(filenames_image[7], cv2.IMREAD_UNCHANGED).astype(np.float32)
    image_normal_z = cv2.imread(filenames_image[8], cv2.IMREAD_UNCHANGED).astype(np.float32)
    image_normal = [image_normal_x, image_normal_y, image_normal_z]
    for i, img in enumerate(image_normal):
        img /= 127
        img -= 1
        image_normal[i] = np.expand_dims(img, 0)

    # cv2.imshow('rgb', image_rgb)
    # cv2.imshow('ir', image_ir)
    # cv2.imshow('depth', image_depth)
    # cv2.imshow('normal z', image_normal[2])
    # cv2.waitKey(0)

    label_array = None
    if self.file_format == "npy":
      label_array = np.load(os.path.join(self.root, filename_labels[0]))
    elif self.file_format == "csv":
      label_array = genfromtxt(os.path.join(self.root, filename_labels[0]), delimiter=',', dtype="float32")
    else:
      print("Unsupported file format " + self.file_format)

    label = Image.fromarray(label_array, 'F')

    # Convert to tensor
    if self.tensor_type=='long':
      label = ToLabel()(label)
    elif self.tensor_type=='float':
      label = ToFloatLabel()(label)

    # Sanitize labels. 
    if self.file_format == "csv":
      label[label != label] = -1

    n_nan = np.count_nonzero(np.isnan(label.numpy()))
    if n_nan > 0:
      print("File " + filename_labels[0] + " produces nan " + str(n_nan))

    label = np.expand_dims(label, 0)

    return (image_rgb, image_ir, image_depth, image_depth_3d[0], image_depth_3d[1], image_depth_3d[2], image_normal[0], image_normal[1], image_normal[2]), label

  def __len__(self):
    return len(self.filenames_img)    



class Transform(object):
  def __init__(self, augment=True, height=512):
    self.augment = augment
    self.height = height

    self.rotation_angle = 5.0
    self.affine_angle = 5.0
    self.shear_angle = 5.0
    # self.crop_ratio = 0.7
    self.gaussian_noise = 0.03 * 255

    self.color_augmentation = ColorJitter(brightness=0.4,
                                          contrast=0.4,
                                          saturation=0.4,
                                          hue=0.06)
    pass

  def transform_augmentation(self, image, flip, rotation, affine_angle, affine_shear):
    # Horizontal flip
    if flip:
        image = functional.hflip(image)
    # Rotate image. 
    image = functional.rotate(image, rotation)
    # Affine transformation
    # image = functional.affine(image, affine_angle, (0,0), affine_shear)   # Affine not available in this pytorch version

    return image


  def __call__(self, input, target):
    # valid_mask = Image.fromarray(np.ones((target.size[1], target.size[0]), dtype=np.uint8),'L')
    # Crop needs to happen here to avoid cropping out all footsteps
    # while True:
    #   # Generate parameters for image transforms
    #   rotation_angle = random.uniform(-self.rotation_angle, self.rotation_angle)  
    #   tan_ang = abs(math.tan(math.radians(rotation_angle)))
    #   y_bound_pix = tan_ang*320
    #   x_bound_pix = tan_ang*240
    #   # crop_val = random.uniform(self.crop_ratio, 1.0-(y_bound_pix/240))
    #   affine_angle = random.uniform(-self.affine_angle, self.affine_angle)
    #   shear_angle = random.uniform(-self.shear_angle, self.shear_angle)
    #   flip = random.random() < 0.5
    #   # img_size = np.array([640, 480]) * crop_val
    #   # hor_pos = int(random.uniform(tan_ang, 1-tan_ang) * (640 - img_size[0]))
    #   # Do other transform.
    #   input_crop = self.transform_augmentation(input, flip, rotation_angle, affine_angle, shear_angle)
    #   target_crop = self.transform_augmentation(target, flip, rotation_angle, affine_angle, shear_angle)
    #   mask_crop = self.transform_augmentation(valid_mask, flip, rotation_angle, affine_angle, shear_angle)
    #   # Do crop
    #   # crop_tuple = (hor_pos, 480 - img_size[1]-y_bound_pix, hor_pos + img_size[0], 480-y_bound_pix)
    #   # input_crop = input_crop.crop(crop_tuple)
    #   # target_crop = target_crop.crop(crop_tuple)
    #   target_test = np.array(target_crop, dtype="float32")
    #   # Make this condition proper for regression where we want > 0.0. Or fix border issues?!
    #   if np.any(target_test != -1):
    #     input = input_crop.resize((640,480))
    #     target = target_crop.resize((640,480))
    #     valid_mask = mask_crop.resize((640,480))
    #     break

    # # Set black parts from transform to invalid.
    # target_np = np.array(target)
    # target_np[np.array(valid_mask)!=1] = -1
    # target = Image.fromarray(target_np)
    # Color transformation
    input_augment = self.color_augmentation(input)
    # Add noise. Since PIL sucks, this is the best way.
    noise = self.gaussian_noise * np.random.randn(input_augment.size[1],input_augment.size[0],len(input_augment.getbands()))
    input_augment = Image.fromarray(np.uint8(np.clip(np.array(input_augment) + noise, 0, 255)))

    return input_augment, input, target
