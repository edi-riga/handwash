#
# This is a generator for combined [rgb, of] input tensors.
# It it based on the Kerase generator code, but useful for the two-stream network.
#

import os
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.keras.preprocessing import image as keras_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops

ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

def merged_dataset_from_directories(rgb_directory,
                                    of_directory,
                                    labels='inferred',
                                    label_mode='int',
                                    class_names=None,
                                    color_mode='rgb',
                                    batch_size=32,
                                    image_size=(256, 256),
                                    shuffle=True,
                                    seed=None,
                                    validation_split=None,
                                    subset=None,
                                    interpolation='bilinear',
                                    follow_links=False,
                                    crop_to_aspect_ratio=False,
                                    **kwargs):
  """Generates a `tf.data.Dataset` from image files in a directory.

  If your directory structure is:

  ```
  main_directory/
  ...class_a/
  ......a_image_1.jpg
  ......a_image_2.jpg
  ...class_b/
  ......b_image_1.jpg
  ......b_image_2.jpg
  ```

  Then calling `image_dataset_from_directory(main_directory, labels='inferred')`
  will return a `tf.data.Dataset` that yields batches of images from
  the subdirectories `class_a` and `class_b`, together with labels
  0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

  Supported image formats: jpeg, png, bmp, gif.
  Animated gifs are truncated to the first frame.

  Args:
    directory: Directory where the data is located.
        If `labels` is "inferred", it should contain
        subdirectories, each containing images for a class.
        Otherwise, the directory structure is ignored.
    labels: Either "inferred"
        (labels are generated from the directory structure),
        None (no labels),
        or a list/tuple of integer labels of the same size as the number of
        image files found in the directory. Labels should be sorted according
        to the alphanumeric order of the image file paths
        (obtained via `os.walk(directory)` in Python).
    label_mode:
        - 'int': means that the labels are encoded as integers
            (e.g. for `sparse_categorical_crossentropy` loss).
        - 'categorical' means that the labels are
            encoded as a categorical vector
            (e.g. for `categorical_crossentropy` loss).
        - 'binary' means that the labels (there can be only 2)
            are encoded as `float32` scalars with values 0 or 1
            (e.g. for `binary_crossentropy`).
        - None (no labels).
    class_names: Only valid if "labels" is "inferred". This is the explict
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
    color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
        Whether the images will be converted to
        have 1, 3, or 4 channels.
    batch_size: Size of the batches of data. Default: 32.
    image_size: Size to resize images to after they are read from disk.
        Defaults to `(256, 256)`.
        Since the pipeline processes batches of images that must all have
        the same size, this must be provided.
    shuffle: Whether to shuffle the data. Default: True.
        If set to False, sorts the data in alphanumeric order.
    seed: Optional random seed for shuffling and transformations.
    validation_split: Optional float between 0 and 1,
        fraction of data to reserve for validation.
    subset: One of "training" or "validation".
        Only used if `validation_split` is set.
    interpolation: String, the interpolation method used when resizing images.
      Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
      `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
    follow_links: Whether to visits subdirectories pointed to by symlinks.
        Defaults to False.
    crop_to_aspect_ratio: If True, resize the images without aspect
      ratio distortion. When the original aspect ratio differs from the target
      aspect ratio, the output image will be cropped so as to return the largest
      possible window in the image (of size `image_size`) that matches
      the target aspect ratio. By default (`crop_to_aspect_ratio=False`),
      aspect ratio may not be preserved.
    **kwargs: Legacy keyword arguments.

  Returns:
    A `tf.data.Dataset` object.
      - If `label_mode` is None, it yields `float32` tensors of shape
        `(batch_size, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
      - Otherwise, it yields a tuple `(images, labels)`, where `images`
        has shape `(batch_size, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.

  Rules regarding labels format:
    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorial`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.

  Rules regarding number of channels in the yielded images:
    - if `color_mode` is `grayscale`,
      there's 1 channel in the image tensors.
    - if `color_mode` is `rgb`,
      there are 3 channel in the image tensors.
    - if `color_mode` is `rgba`,
      there are 4 channel in the image tensors.
  """
  if 'smart_resize' in kwargs:
    crop_to_aspect_ratio = kwargs.pop('smart_resize')
  if kwargs:
    raise TypeError(f'Unknown keywords argument(s): {tuple(kwargs.keys())}')
  if labels not in ('inferred', None):
    if not isinstance(labels, (list, tuple)):
      raise ValueError(
          '`labels` argument should be a list/tuple of integer labels, of '
          'the same size as the number of image files in the target '
          'directory. If you wish to infer the labels from the subdirectory '
          'names in the target directory, pass `labels="inferred"`. '
          'If you wish to get a dataset that only contains images '
          '(no labels), pass `label_mode=None`.')
    if class_names:
      raise ValueError('You can only pass `class_names` if the labels are '
                       'inferred from the subdirectory names in the target '
                       'directory (`labels="inferred"`).')
  if label_mode not in {'int', 'categorical', 'binary', None}:
    raise ValueError(
        '`label_mode` argument must be one of "int", "categorical", "binary", '
        'or None. Received: %s' % (label_mode,))
  if labels is None or label_mode is None:
    labels = None
    label_mode = None
  if color_mode == 'rgb':
    num_channels = 3
  elif color_mode == 'rgba':
    num_channels = 4
  elif color_mode == 'grayscale':
    num_channels = 1
  else:
    raise ValueError(
        '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
        'Received: %s' % (color_mode,))
  interpolation = image_preprocessing.get_interpolation(interpolation)
  dataset_utils.check_validation_split_arg(
      validation_split, subset, shuffle, seed)

  if seed is None:
    seed = np.random.randint(1e6)
  of_image_paths, labels, class_names = dataset_utils.index_directory(
      of_directory,
      labels,
      formats=ALLOWLIST_FORMATS,
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

  dir_of = os.path.abspath(of_directory.rstrip("/"))
  dir_rgb = os.path.abspath(rgb_directory.rstrip("/"))
  #print("dir_of=", dir_of)
  #print("dir_rgb=", dir_rgb)
  rgb_image_paths = []
  for of_path in of_image_paths:
      rgb_path = of_path.replace(dir_of, dir_rgb)
      # print("add path", rgb_path, "created from", of_path)
      if not os.access(rgb_path, os.R_OK):
          raise TypeError(f'no such file: {rgb_path}')
      rgb_image_paths.append(rgb_path)

  if label_mode == 'binary' and len(class_names) != 2:
    raise ValueError(
        'When passing `label_mode="binary", there must exactly 2 classes. '
        'Found the following classes: %s' % (class_names,))

  of_image_paths, labels = dataset_utils.get_training_or_validation_split(
      of_image_paths, labels, validation_split, subset)
  rgb_image_paths, _ = dataset_utils.get_training_or_validation_split(
      rgb_image_paths, labels, validation_split, subset)

  if not of_image_paths:
    raise ValueError('No OF images found.')
  if not rgb_image_paths:
    raise ValueError('No RGB images found.')

  num_classes=len(class_names)

  of_dataset = paths_and_labels_to_dataset(
      image_paths=of_image_paths,
      image_size=image_size,
      num_channels=num_channels,
      num_classes=num_classes,
      interpolation=interpolation,
      crop_to_aspect_ratio=crop_to_aspect_ratio)

  rgb_dataset = paths_and_labels_to_dataset(
      image_paths=rgb_image_paths,
      image_size=image_size,
      num_channels=num_channels,
      num_classes=num_classes,
      interpolation=interpolation,
      crop_to_aspect_ratio=crop_to_aspect_ratio)

  img_dataset = dataset_ops.Dataset.zip((rgb_dataset, of_dataset))

  if label_mode:
    label_dataset = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    dataset = dataset_ops.Dataset.zip((img_dataset, label_dataset))
  else:
    dataset = img_dataset

  if shuffle:
    # Shuffle locally at each iteration
    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
  dataset = dataset.batch(batch_size)
  # Users may need to reference `class_names`.
  dataset.class_names = class_names
  # Include file paths for images as attribute.
  dataset.file_paths = of_image_paths
  return dataset


def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                num_channels,
                                num_classes,
                                interpolation,
                                crop_to_aspect_ratio=False):
  """Constructs a dataset of images and labels."""
  # TODO(fchollet): consider making num_parallel_calls settable
  path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
  args = (image_size, num_channels, interpolation, crop_to_aspect_ratio)
  img_ds = path_ds.map(
      lambda x: load_image(x, *args))
  return img_ds

def load_image(path, image_size, num_channels, interpolation,
               crop_to_aspect_ratio=False):
  """Load an image from a path and resize it."""
  img = io_ops.read_file(path)
  img = image_ops.decode_image(
      img, channels=num_channels, expand_animations=False)
  if crop_to_aspect_ratio:
    img = keras_image_ops.smart_resize(img, image_size,
                                       interpolation=interpolation)
  else:
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
  img.set_shape((image_size[0], image_size[1], num_channels))
  return img
