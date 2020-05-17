import tensorflow as tf

def get_weighted_boxes_and_landmarks(scores, bboxes_and_landmarks, mask):
    """Calculating weighted mean of given bboxes and landmarks according to the mask.
    inputs:
        scores = (total_bboxes, [probability])
        bboxes_and_landmarks = (total_bboxes, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])
        mask = (total_bboxes,)

    outputs:
        weighted_bbox_and_landmark = (1, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])
    """
    selected_scores = scores[mask]
    selected_bboxes_and_landmarks = bboxes_and_landmarks[mask]
    weighted_sum = tf.reduce_sum(selected_bboxes_and_landmarks * selected_scores, 0)
    sum_selected_scores = tf.reduce_sum(selected_scores, 0)
    sum_selected_scores = tf.where(tf.equal(sum_selected_scores, 0.0), 1.0, sum_selected_scores)
    return tf.expand_dims(weighted_sum / sum_selected_scores, 0)

def weighted_suppression_body(counter, iou_threshold, scores, bboxes_and_landmarks, weighted_suppressed_data):
    """Weighted mean suppression algorithm while body.
    inputs:
        counter = while body counter
        iou_threshold = threshold value for overlapping bounding boxes
        scores = (total_bboxes, [probability])
        bboxes_and_landmarks = (total_bboxes, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])
        weighted_suppressed_data = (M, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])

    outputs:
        counter = while body counter
        iou_threshold = threshold value for overlapping bounding boxes
        scores = (total_bboxes - N, [probability])
        bboxes_and_landmarks = (total_bboxes - N, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])
        weighted_suppressed_data = (M + 1, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])
    """
    counter = tf.add(counter, 1)
    first_box = bboxes_and_landmarks[0, 0:4]
    iou_map = generate_iou_map(first_box, bboxes_and_landmarks[..., 0:4], transpose_perm=[1, 0])
    overlapped_mask = tf.reshape(tf.greater(iou_map, iou_threshold), (-1,))
    weighted_bbox_and_landmark = get_weighted_boxes_and_landmarks(scores, bboxes_and_landmarks, overlapped_mask)
    weighted_suppressed_data = tf.concat([weighted_suppressed_data, weighted_bbox_and_landmark], axis=0)
    not_overlapped_mask = tf.logical_not(overlapped_mask)
    scores = scores[not_overlapped_mask]
    bboxes_and_landmarks = bboxes_and_landmarks[not_overlapped_mask]
    return counter, iou_threshold, scores, bboxes_and_landmarks, weighted_suppressed_data

def weighted_suppression(scores, bboxes_and_landmarks, max_total_size=50, score_threshold=0.75, iou_threshold=0.3):
    """Blazeface weighted mean suppression algorithm.
    inputs:
        scores = (total_bboxes, [probability])
        bboxes_and_landmarks = (total_bboxes, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])
        max_total_size = maximum returned bounding boxes and landmarks
        score_threshold = threshold value for bounding boxes and landmarks selection
        iou_threshold = threshold value for overlapping bounding boxes

    outputs:
        weighted_bboxes_and_landmarks = (dynamic_size, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])
    """
    score_mask = tf.squeeze(tf.greater(scores, score_threshold), -1)
    scores = scores[score_mask]
    bboxes_and_landmarks = bboxes_and_landmarks[score_mask]
    sorted_indices = tf.argsort(scores, axis=0, direction="DESCENDING")
    sorted_scores = tf.gather_nd(scores, sorted_indices)
    sorted_bboxes_and_landmarks = tf.gather_nd(bboxes_and_landmarks, sorted_indices)
    counter = tf.constant(0, tf.int32)
    weighted_data = tf.zeros(tf.shape(bboxes_and_landmarks[0:1]), dtype=tf.float32)
    cond = lambda counter, iou_threshold, scores, data, weighted: tf.logical_and(tf.less(counter, max_total_size), tf.greater(tf.shape(scores)[0], 0))
    _, _, _, _, weighted_data = tf.while_loop(cond, weighted_suppression_body,
                                          [counter, iou_threshold, sorted_scores, sorted_bboxes_and_landmarks, weighted_data])
    #
    return weighted_data[1:]

def non_max_suppression(pred_bboxes, pred_labels, **kwargs):
    """Applying non maximum suppression.
    Details could be found on tensorflow documentation.
    https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    inputs:
        pred_bboxes = (batch_size, total_bboxes, total_labels, [y1, x1, y2, x2])
            total_labels should be 1 for binary operations like in rpn
        pred_labels = (batch_size, total_bboxes, total_labels)
        **kwargs = other parameters

    outputs:
        nms_boxes = (batch_size, max_detections, [y1, x1, y2, x2])
        nmsed_scores = (batch_size, max_detections)
        nmsed_classes = (batch_size, max_detections)
        valid_detections = (batch_size)
            Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid.
            The rest of the entries are zero paddings.
    """
    return tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        **kwargs
    )

def generate_iou_map(bboxes, gt_boxes, transpose_perm=[0, 2, 1]):
    """Calculating intersection over union values for each ground truth boxes in a dynamic manner.
    It is supported from 1d to 3d dimensions for bounding boxes.
    Even if bboxes have different rank from gt_boxes it should be work.
    inputs:
        bboxes = (dynamic_dimension, [y1, x1, y2, x2])
        gt_boxes = (dynamic_dimension, [y1, x1, y2, x2])
        transpose_perm = (transpose_perm_order)
            for 3d gt_boxes => [0, 2, 1]

    outputs:
        iou_map = (dynamic_dimension, total_gt_boxes)
            same rank with the gt_boxes
    """
    gt_rank = tf.rank(gt_boxes)
    gt_expand_axis = gt_rank - 2
    #
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)
    # Calculate bbox and ground truth boxes areas
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    #
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, transpose_perm))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, transpose_perm))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, transpose_perm))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, transpose_perm))
    ### Calculate intersection area
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    ### Calculate union area
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, gt_expand_axis) - intersection_area)
    # Intersection over Union
    return intersection_area / union_area

def get_bboxes_and_landmarks_from_deltas(prior_boxes, deltas):
    """Calculating bounding boxes and landmarks for given delta values.
    inputs:
        prior_boxes = (total_bboxes, [center_x, center_y, width, height])
        deltas = (batch_size, total_bboxes, [delta_bbox_y, delta_bbox_x, delta_bbox_h, delta_bbox_w, delta_landmark_x0, delta_landmark_y0, ..., delta_landmark_xN, delta_landmark_yN])

    outputs:
        bboxes_and_landmarks = (batch_size, total_bboxes, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])
    """
    #
    bbox_width = deltas[..., 3] * prior_boxes[..., 2]
    bbox_height = deltas[..., 2] * prior_boxes[..., 3]
    bbox_ctr_x = (deltas[..., 1] * prior_boxes[..., 2]) + prior_boxes[..., 0]
    bbox_ctr_y = (deltas[..., 0] * prior_boxes[..., 3]) + prior_boxes[..., 1]
    #
    y1 = bbox_ctr_y - (0.5 * bbox_height)
    x1 = bbox_ctr_x - (0.5 * bbox_width)
    y2 = bbox_height + y1
    x2 = bbox_width + x1
    #
    total_landmarks = tf.shape(deltas[..., 4:])[-1] // 2
    xy_pairs = tf.tile(prior_boxes[..., 0:2], (1, total_landmarks))
    wh_pairs = tf.tile(prior_boxes[..., 2:4], (1, total_landmarks))
    landmarks = (deltas[..., 4:] * wh_pairs) + xy_pairs
    #
    return tf.concat([tf.stack([y1, x1, y2, x2], axis=-1), landmarks], -1)

def get_deltas_from_bboxes_and_landmarks(prior_boxes, bboxes_and_landmarks):
    """Calculating bounding box and landmark deltas for given ground truth boxes and landmarks.
    inputs:
        prior_boxes = (total_bboxes, [center_x, center_y, width, height])
        bboxes_and_landmarks = (batch_size, total_bboxes, [y1, x1, y2, x2, landmark_x0, landmark_y0, ..., landmark_xN, landmark_yN])

    outputs:
        deltas = (batch_size, total_bboxes, [delta_bbox_y, delta_bbox_x, delta_bbox_h, delta_bbox_w, delta_landmark_x0, delta_landmark_y0, ..., delta_landmark_xN, delta_landmark_yN])
    """
    #
    gt_width = bboxes_and_landmarks[..., 3] - bboxes_and_landmarks[..., 1]
    gt_height = bboxes_and_landmarks[..., 2] - bboxes_and_landmarks[..., 0]
    gt_ctr_x = bboxes_and_landmarks[..., 1] + 0.5 * gt_width
    gt_ctr_y = bboxes_and_landmarks[..., 0] + 0.5 * gt_height
    #
    delta_x = (gt_ctr_x - prior_boxes[..., 0]) / prior_boxes[..., 2]
    delta_y = (gt_ctr_y - prior_boxes[..., 1]) / prior_boxes[..., 3]
    delta_w = gt_width / prior_boxes[..., 2]
    delta_h = gt_height / prior_boxes[..., 3]
    #
    total_landmarks = tf.shape(bboxes_and_landmarks[..., 4:])[-1] // 2
    xy_pairs = tf.tile(prior_boxes[..., 0:2], (1, total_landmarks))
    wh_pairs = tf.tile(prior_boxes[..., 2:4], (1, total_landmarks))
    landmark_deltas = (bboxes_and_landmarks[..., 4:] - xy_pairs) / wh_pairs
    #
    return tf.concat([tf.stack([delta_y, delta_x, delta_h, delta_w], -1), landmark_deltas], -1)

def get_scale_for_nth_feature_map(k, m=4, scale_min=0.1484375, scale_max=0.75):
    """Calculating scale value for nth feature map using the given method in the paper.
    inputs:
        k = nth feature map for scale calculation
        m = length of all using feature maps for detections, 6 for ssd300, 4 for blazeface

    outputs:
        scale = calculated scale value for given index
    """
    return scale_min + ((scale_max - scale_min) / (m - 1)) * (k - 1)

def get_wh_pairs(aspect_ratios, feature_map_index, total_feature_map):
    """Generating height and width pairs for different aspect ratios and feature map shapes.
    inputs:
        aspect_ratios = for all feature map shapes + 1 for ratio 1
        feature_map_index = nth feature maps for scale calculation
        total_feature_map = length of all using feature map for detections, 6 for ssd300

    outputs:
        wh_pairs = [(width1, height1), ..., (widthN, heightN)]
    """
    current_scale = get_scale_for_nth_feature_map(feature_map_index, m=total_feature_map)
    next_scale = get_scale_for_nth_feature_map(feature_map_index + 1, m=total_feature_map)
    wh_pairs = []
    for aspect_ratio in aspect_ratios:
        height = current_scale / tf.sqrt(aspect_ratio)
        width = current_scale * tf.sqrt(aspect_ratio)
        wh_pairs.append([width, height])
    # 1 extra pair for ratio 1
    height = width = tf.sqrt(current_scale * next_scale)
    wh_pairs.append([width, height])
    return tf.cast(wh_pairs, dtype=tf.float32)

def generate_prior_boxes(feature_map_shapes, aspect_ratios):
    """Generating top left prior boxes for given stride, height and width pairs of different aspect ratios.
    These prior boxes same with the anchors in Faster-RCNN.
    inputs:
        feature_map_shapes = for all feature map output size
        aspect_ratios = for all feature map shapes + 1 for ratio 1

    outputs:
        prior_boxes = (total_bboxes, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
    """
    prior_boxes = []
    for i, feature_map_shape in enumerate(feature_map_shapes):
        wh_pairs = get_wh_pairs(aspect_ratios[i], i+1, len(feature_map_shapes))
        #
        stride = 1 / feature_map_shape
        grid_coords = tf.cast(tf.range(0, feature_map_shape) / feature_map_shape + stride / 2, dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords)
        flat_grid_x, flat_grid_y = tf.reshape(grid_x, (-1, )), tf.reshape(grid_y, (-1, ))
        #
        grid_map = tf.stack([flat_grid_x, flat_grid_y], axis=-1)
        grid_map = tf.pad(grid_map, ((0,0), (0,2)))
        wh_pairs = tf.pad(wh_pairs, ((0,0), (2,0)))
        #
        prior_boxes_for_feature_map = tf.reshape(wh_pairs, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
        prior_boxes_for_feature_map = tf.reshape(prior_boxes_for_feature_map, (-1, 4))
        #
        prior_boxes.append(prior_boxes_for_feature_map)
    prior_boxes = tf.concat(prior_boxes, axis=0)
    return tf.clip_by_value(prior_boxes, 0, 1)

def convert_bboxes_to_xywh(bboxes):
    """Converting bounding boxes to center x, y and width height format.
    inputs:
        bboxes = (M x N, [y1, x1, y2, x2])

    outputs:
        xywh = (M x N, [center_x, center_y, width, height])
    """
    width = bboxes[..., 3] - bboxes[..., 1]
    height = bboxes[..., 2] - bboxes[..., 0]
    center_x = bboxes[..., 1] + 0.5 * width
    center_y = bboxes[..., 0] + 0.5 * height
    xywh = tf.stack([center_x, center_y, width, height], axis=-1)
    return tf.clip_by_value(xywh, 0, 1)

def convert_xywh_to_bboxes(xywh):
    """Converting center x, y and width height format to bounding boxes.
    inputs:
        xywh = (M x N, [center_x, center_y, width, height])

    outputs:
        bboxes = (M x N, [y1, x1, y2, x2])
    """
    y1 = xywh[..., 1] - (0.5 * xywh[..., 3])
    x1 = xywh[..., 0] - (0.5 * xywh[..., 2])
    y2 = xywh[..., 3] + y1
    x2 = xywh[..., 2] + x1
    bboxes = tf.stack([y1, x1, y2, x2], axis=-1)
    return tf.clip_by_value(bboxes, 0, 1)

def normalize_bboxes(bboxes, height, width):
    """Normalizing bounding boxes.
    inputs:
        bboxes = (M x N, [y1, x1, y2, x2])
        height = image height
        width = image width

    outputs:
        normalized_bboxes = (M x N, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    y1 = bboxes[..., 0] / height
    x1 = bboxes[..., 1] / width
    y2 = bboxes[..., 2] / height
    x2 = bboxes[..., 3] / width
    return tf.stack([y1, x1, y2, x2], axis=-1)

def denormalize_bboxes(bboxes, height, width):
    """Denormalizing bounding boxes.
    inputs:
        bboxes = (M x N, [y1, x1, y2, x2])
            in normalized form [0, 1]
        height = image height
        width = image width

    outputs:
        denormalized_bboxes = (M x N, [y1, x1, y2, x2])
    """
    y1 = bboxes[..., 0] * height
    x1 = bboxes[..., 1] * width
    y2 = bboxes[..., 2] * height
    x2 = bboxes[..., 3] * width
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))
