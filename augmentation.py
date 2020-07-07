import tensorflow as tf
from utils import bbox_utils, landmark_utils

def apply(img, gt_boxes, gt_landmarks):
    """Randomly applying data augmentation methods to image and ground truth boxes.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
            in normalized form [0, 1]
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
            in normalized form [0, 1]
    outputs:
        modified_img = (final_height, final_width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
            in normalized form [0, 1]
        modified_gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
            in normalized form [0, 1]
    """
    # Color operations
    # Randomly change hue, saturation, brightness and contrast of image
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation]
    # Geometric operations
    # Randomly sample a patch image and ground truth boxes
    geometric_methods = [patch]
    #
    for augmentation_method in geometric_methods + color_methods:
        img, gt_boxes, gt_landmarks = randomly_apply_operation(augmentation_method, img, gt_boxes, gt_landmarks)
    #
    img = tf.clip_by_value(img, 0., 1.)
    return img, gt_boxes, gt_landmarks

def get_random_bool():
    """Generating random boolean.
    outputs:
        random boolean 0d tensor
    """
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)

def randomly_apply_operation(operation, img, gt_boxes, gt_landmarks, *args):
    """Randomly applying given method to image and ground truth boxes.
    inputs:
        operation = callable method
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    outputs:
        modified_or_not_img = (final_height, final_width, depth)
        modified_or_not_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        modified_or_not_gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    """
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes, gt_landmarks, *args),
        lambda: (img, gt_boxes, gt_landmarks)
    )

def random_brightness(img, gt_boxes, gt_landmarks, max_delta=0.12):
    """Randomly change brightness of the image.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    outputs:
        modified_img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    """
    return tf.image.random_brightness(img, max_delta), gt_boxes, gt_landmarks

def random_contrast(img, gt_boxes, gt_landmarks, lower=0.5, upper=1.5):
    """Randomly change contrast of the image.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    outputs:
        modified_img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    """
    return tf.image.random_contrast(img, lower, upper), gt_boxes, gt_landmarks

def random_hue(img, gt_boxes, gt_landmarks, max_delta=0.08):
    """Randomly change hue of the image.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    outputs:
        modified_img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    """
    return tf.image.random_hue(img, max_delta), gt_boxes, gt_landmarks

def random_saturation(img, gt_boxes, gt_landmarks, lower=0.5, upper=1.5):
    """Randomly change saturation of the image.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    outputs:
        modified_img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    """
    return tf.image.random_saturation(img, lower, upper), gt_boxes, gt_landmarks

##############################################################################
## Sample patch start
##############################################################################

def get_random_min_overlap():
    """Generating random minimum overlap value.
    outputs:
        min_overlap = random minimum overlap value 0d tensor
    """
    overlaps = tf.constant([0.1, 0.3, 0.5, 0.7, 0.9], dtype=tf.float32)
    i = tf.random.uniform((), minval=0, maxval=tf.shape(overlaps)[0], dtype=tf.int32)
    return overlaps[i]

def expand_image(img, gt_boxes, gt_landmarks, height, width):
    """Randomly expanding image and adjusting ground truth object coordinates.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
        height = height of the image
        width = width of the image
    outputs:
        modified_img = (final_height, final_width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        modified_gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
    """
    expansion_ratio = tf.random.uniform((), minval=1, maxval=4, dtype=tf.float32)
    final_height, final_width = tf.round(height * expansion_ratio), tf.round(width * expansion_ratio)
    pad_left = tf.round(tf.random.uniform((), minval=0, maxval=final_width - width, dtype=tf.float32))
    pad_top = tf.round(tf.random.uniform((), minval=0, maxval=final_height - height, dtype=tf.float32))
    pad_right = final_width - (width + pad_left)
    pad_bottom = final_height - (height + pad_top)
    #
    mean, _ = tf.nn.moments(img, [0, 1])
    expanded_image = tf.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), constant_values=-1)
    expanded_image = tf.where(expanded_image == -1, mean, expanded_image)
    #
    min_max = tf.stack([-pad_top, -pad_left, pad_bottom+height, pad_right+width], -1) / [height, width, height, width]
    modified_gt_boxes = bbox_utils.renormalize_bboxes_with_min_max(gt_boxes, min_max)
    modified_gt_landmarks = landmark_utils.renormalize_landmarks_with_min_max(gt_landmarks, min_max)
    #
    return expanded_image, modified_gt_boxes, modified_gt_landmarks

def patch(img, gt_boxes, gt_landmarks):
    """Generating random patch and adjusting image and ground truth objects to this patch.
    After this operation some of the ground truth boxes / objects could be removed from the image.
    However, these objects are not excluded from the output, only the coordinates are changed as zero.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
            in normalized form [0, 1]
        gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
            in normalized form [0, 1]
    outputs:
        modified_img = (final_height, final_width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
            in normalized form [0, 1]
        modified_gt_landmarks = (ground_truth_object_count, total_landmarks, [x, y])
            in normalized form [0, 1]
    """
    img_shape = tf.cast(tf.shape(img), dtype=tf.float32)
    org_height, org_width = img_shape[0], img_shape[1]
    # Randomly expand image and adjust bounding boxes
    img, gt_boxes, gt_landmarks = randomly_apply_operation(expand_image, img, gt_boxes, gt_landmarks, org_height, org_width)
    # Get random minimum overlap value
    min_overlap = get_random_min_overlap()
    #
    begin, size, new_boundaries = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=tf.expand_dims(gt_boxes, 0),
        min_object_covered=min_overlap)
    #
    img = tf.slice(img, begin, size)
    img = tf.image.resize(img, (org_height, org_width))
    gt_boxes = bbox_utils.renormalize_bboxes_with_min_max(gt_boxes, new_boundaries[0, 0])
    gt_landmarks = landmark_utils.renormalize_landmarks_with_min_max(gt_landmarks, new_boundaries[0, 0])
    #
    return img, gt_boxes, gt_landmarks
