import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from utils import landmark_utils
import numpy as np

def filter_landmarks(landmarks):
    """Filtering landmark from 68 points to 6 points for blazeface.
    inputs:
        landmarks = (M x N, [x, y])

    outputs:
        filtered_landmarks = (M x 6, [x, y])
    """
    # Right eye
    right_eye_coords = tf.reduce_mean(landmarks[..., 36:42, :], -2)
    # Left eye
    left_eye_coords = tf.reduce_mean(landmarks[..., 42:48, :], -2)
    # Right ear
    right_ear_coords = tf.reduce_mean(landmarks[..., 0:2, :], -2)
    # Left ear
    left_ear_coords = tf.reduce_mean(landmarks[..., 15:17, :], -2)
    # Nose
    nose_coords = tf.reduce_mean(landmarks[..., 27:36, :], -2)
    # Mouth
    mouth_coords = tf.reduce_mean(landmarks[..., 48:68, :], -2)
    return tf.stack([
        right_eye_coords,
        left_eye_coords,
        right_ear_coords,
        left_ear_coords,
        nose_coords,
        mouth_coords,
    ], -2)

def generate_bboxes_from_landmarks(landmarks):
    """Generating bounding boxes from landmarks.
    inputs:
        landmarks = (M x N, [x, y])

    outputs:
        bboxes = (M, [y1, x1, y2, x2])
    """
    padding = 5e-3
    x1 = tf.reduce_min(landmarks[..., 0], -1) - padding
    x2 = tf.reduce_max(landmarks[..., 0], -1) + padding
    y1 = tf.reduce_min(landmarks[..., 1], -1) - padding
    y2 = tf.reduce_max(landmarks[..., 1], -1) + padding
    #
    gt_boxes = tf.stack([y1, x1, y2, x2], -1)
    return tf.clip_by_value(gt_boxes, 0, 1)

def preprocessing(image_data, final_height, final_width, augmentation_fn=None):
    """Image resizing operation handled before batch operations.
    inputs:
        image_data = tensorflow dataset image_data
        final_height = final image height after resizing
        final_width = final image width after resizing

    outputs:
        img = (final_height, final_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_landmarks = (gt_box_size, 6, [x, y])
    """
    img = image_data["image"]
    img = tf.image.convert_image_dtype(img, tf.float32)
    gt_landmarks = tf.expand_dims(image_data["landmarks_2d"], 0)
    gt_boxes = generate_bboxes_from_landmarks(gt_landmarks)
    gt_landmarks = filter_landmarks(gt_landmarks)
    img = tf.image.resize(img, (final_height, final_width))
    if augmentation_fn:
        img, gt_boxes, gt_landmarks = augmentation_fn(img, gt_boxes, gt_landmarks)
    return img, gt_boxes, gt_landmarks

def get_dataset(name, split, data_dir="~/tensorflow_datasets"):
    """Get tensorflow dataset split and info.
    inputs:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets

    outputs:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info

def get_total_item_size(info, split):
    """Get total item size for given split.
    inputs:
        info = tensorflow dataset info
        split = data split string, should be one of ["train", "validation", "test"]

    outputs:
        total_item_size = number of total items
    """
    return info.splits[split].num_examples

def get_labels(info):
    """Get label names list.
    inputs:
        info = tensorflow dataset info

    outputs:
        labels = [labels list]
    """
    return info.features["labels"].names

def get_custom_imgs(custom_image_path):
    """Generating a list of images for given path.
    inputs:
        custom_image_path = folder of the custom images
    outputs:
        custom image list = [path1, path2]
    """
    img_paths = []
    for path, dir, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_paths.append(os.path.join(path, filename))
        break
    return img_paths

def custom_data_generator(img_paths, final_height, final_width):
    """Yielding custom entities as dataset.
    inputs:
        img_paths = custom image paths
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, depth)
        dummy_gt_boxes = (None, None)
        dummy_gt_labels = (None, )
    """
    for img_path in img_paths:
        image = Image.open(img_path)
        resized_image = image.resize((final_width, final_height), Image.LANCZOS)
        img = np.array(resized_image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        yield img, tf.constant([[]], dtype=tf.float32), tf.constant([[[]]], dtype=tf.float32)

def get_data_types():
    """Generating dataset parameter dtypes for tensorflow datasets.
    outputs:
        dtypes = output dtypes for (images, ground truth boxes, ground truth landmarks)
    """
    return (tf.float32, tf.float32, tf.float32)

def get_data_shapes():
    """Generating dataset parameter shapes for tensorflow datasets.
    outputs:
        shapes = output shapes for (images, ground truth boxes, ground truth landmarks)
    """
    return ([None, None, None], [None, None], [None, None, None])

def get_padding_values():
    """Generating padding values for missing values in batch for tensorflow datasets.
    outputs:
        paddings = padding values with dtypes for (images, ground truth boxes, ground truth landmarks)
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(0, tf.float32))
