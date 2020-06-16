import tensorflow as tf
import math
from . import bbox_utils

def get_hyper_params(**kwargs):
    """Generating hyper params in a dynamic way.
    inputs:
        **kwargs = any value could be updated in the hyper_params

    outputs:
        hyper_params = dictionary
    """
    hyper_params = {
        "img_size": 128,
        "feature_map_shapes": [16, 8, 8, 8],
        "aspect_ratios": [[1.], [1.], [1.], [1.]],
        "detections_per_layer": [2, 6],
        "total_landmarks": 6,
        "iou_threshold": 0.5,
        "neg_pos_ratio": 3,
        "loc_loss_alpha": 1,
        "variances": [0.1, 0.1, 0.2, 0.2],
    }
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    #
    return hyper_params

def scheduler(epoch):
    """Generating learning rate value for a given epoch.
    inputs:
        epoch = number of current epoch

    outputs:
        learning_rate = float learning rate value
    """
    if epoch < 150:
        return 1e-3
    elif epoch < 200:
        return 1e-4
    else:
        return 1e-5

def get_step_size(total_items, batch_size):
    """Get step size for given total item size and batch size.
    inputs:
        total_items = number of total items
        batch_size = number of batch size during training or validation

    outputs:
        step_size = number of step size for model training
    """
    return math.ceil(total_items / batch_size)

def generator(dataset, prior_boxes, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        prior_boxes = (total_bboxes, [center_x, center_y, width, height])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_landmarks = image_data
            actual_deltas, actual_labels = calculate_actual_outputs(prior_boxes, gt_boxes, gt_landmarks, hyper_params)
            yield img, (actual_deltas, actual_labels)

def calculate_actual_outputs(prior_boxes, gt_boxes, gt_landmarks, hyper_params):
    """Calculate ssd actual output values.
    Batch operations supported.
    inputs:
        prior_boxes = (total_bboxes, [center_x, center_y, width, height])
            these values in normalized format between [0, 1]
        gt_boxes = (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_landmarks = (batch_size, gt_box_size, total_landmarks, [x, y])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        actual_deltas = (batch_size, total_bboxes, [delta_bbox_y, delta_bbox_x, delta_bbox_h, delta_bbox_w, delta_landmark_x0, delta_landmark_y0, ..., delta_landmark_xN, delta_landmark_yN])
        actual_labels = (batch_size, total_bboxes, [1 or 0])
    """
    batch_size = tf.shape(gt_boxes)[0]
    iou_threshold = hyper_params["iou_threshold"]
    variances = hyper_params["variances"]
    total_landmarks = hyper_params["total_landmarks"]
    landmark_variances = total_landmarks * variances[0:2]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = bbox_utils.generate_iou_map(bbox_utils.convert_xywh_to_bboxes(prior_boxes), gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_cond = tf.greater(merged_iou_map, iou_threshold)
    #
    gt_landmarks = tf.reshape(gt_landmarks, (batch_size, -1, total_landmarks * 2))
    gt_boxes_and_landmarks = tf.concat([gt_boxes, gt_landmarks], -1)
    gt_boxes_and_landmarks_map = tf.gather(gt_boxes_and_landmarks, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes_and_landmarks = tf.where(tf.expand_dims(pos_cond, -1), gt_boxes_and_landmarks_map, tf.zeros_like(gt_boxes_and_landmarks_map))
    actual_deltas = bbox_utils.get_deltas_from_bboxes_and_landmarks(prior_boxes, expanded_gt_boxes_and_landmarks) / (variances + landmark_variances)
    #
    actual_labels = tf.expand_dims(tf.cast(pos_cond, dtype=tf.float32), -1)
    #
    return actual_deltas, actual_labels
