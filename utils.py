"""Utility functions for reading and saving images."""

import glob
import numpy as np
import scipy
import scipy.misc
import cv2
import imageio
from training.misc import adjust_dynamic_range


def preparing_data(im_path, img_type):
    """
    read images from the given path, and transform images from [0, 255] to [-1., 1.]

    return image shape: [N, C, H, W]
    """
    images = sorted(glob.glob(im_path + '/*' + img_type))
    images_name = []
    input_images = []
    for im_name in images:
        input_images.append(cv2.imread(im_name)[:, :, ::-1])
        images_name.append(im_name.split('/')[-1].split('.')[0])
    input_images = np.asarray(input_images)
    input_images = adjust_dynamic_range(input_images.astype(np.float32), [0, 255], [-1., 1.])
    input_images = input_images.transpose(0, 3, 1, 2)
    return input_images, images_name


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """
    transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    """
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def imwrite(image, path):
    """ save an [-1.0, 1.0] image """

    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    #return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))
    return imageio.imwrite(path, to_range(image, 0, 255, np.uint8))

def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img

def project_boundary(primal, *args):
  """Projects the primal boundary onto condition boundaries.

  The function is used for conditional manipulation, where the projected vector
  will be subscribed from the normal direction of the original boundary. Here,
  all input boundaries are supposed to have already been normalized to unit
  norm, and with same shape [1, latent_space_dim].

  NOTE: For now, at most two condition boundaries are supported.

  Args:
    primal: The primal boundary.
    *args: Other boundaries as conditions.

  Returns:
    A projected boundary (also normalized to unit norm), which is orthogonal to
      all condition boundaries.

  Raises:
    NotImplementedError
/liaGAN_experiments/dnnlib/
Name
Last Modified

: If there are more than two condition boundaries.
  """
  if len(args) > 2:
    raise NotImplementedError(f'This function supports projecting with at most '
                              f'two conditions.')
  assert len(primal.shape) == 2 and primal.shape[0] == 1

  if not args:
    return primal
  if len(args) == 1:
    cond = args[0]
    assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
            cond.shape[1] == primal.shape[1])
    new = primal - primal.dot(cond.T) * cond
    return new / np.linalg.norm(new)
  if len(args) == 2:
    cond_1 = args[0]
    cond_2 = args[1]
    assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
            cond_1.shape[1] == primal.shape[1])
    assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
            cond_2.shape[1] == primal.shape[1])
    primal_cond_1 = primal.dot(cond_1.T)
    primal_cond_2 = primal.dot(cond_2.T)
    cond_1_cond_2 = cond_1.dot(cond_2.T)
    alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    new = primal - alpha * cond_1 - beta * cond_2
    return new / np.linalg.norm(new)

  raise NotImplementedError


def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):
  """Manipulates the given latent code with respect to a particular boundary.

  Basically, this function takes a latent code and a boundary as inputs, and
  outputs a collection of manipulated latent codes. For example, let `steps` to
  be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
  `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
  with shape [10, latent_space_dim]. The first output latent code is
  `start_distance` away from the given `boundary`, while the last output latent
  code is `end_distance` away from the given `boundary`. Remaining latent codes
  are linearly interpolated.

  Input `latent_code` can also be with shape [1, num_layers, latent_space_dim]
  to support W+ space in Style GAN. In this case, all features in W+ space will
  be manipulated same as each other. Accordingly, the output will be with shape
  [10, num_layers, latent_space_dim].

  NOTE: Distance is sign sensitive.

  Args:
    latent_code: The input latent code for manipulation.
    boundary: The semantic boundary as reference.
    start_distance: The distance to the boundary where the manipulation starts.
      (default: -3.0)
    end_distance: The distance to the boundary where the manipulation ends.
      (default: 3.0)
    steps: Number of steps to move the latent code from start position to end
      position. (default: 10)
  """
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1)
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                   f'W+ space in Style GAN!\n'
                   f'But {latent_code.shape} is received.')

