import tensorflow as tf

def renormalize_landmarks_with_min_max(landmarks, min_max):
    """Renormalizing given bounding boxes to the new boundaries.
    r = (x - min) / (max - min)
    outputs:
        landmarks = (total_count, total_landmarks, [x, y])
        min_max = ([y_min, x_min, y_max, x_max])
    """
    y_min, x_min, y_max, x_max = tf.split(min_max, 4)
    renomalized_landmarks = landmarks - tf.concat([x_min, y_min], -1)
    renomalized_landmarks /= tf.concat([x_max-x_min, y_max-y_min], -1)
    return tf.clip_by_value(renomalized_landmarks, 0, 1)

def normalize_landmarks(landmarks, height, width):
    """Normalizing landmarks.
    inputs:
        landmarks = (M, N, [x, y])
        height = image height
        width = image width

    outputs:
        normalized_landmarks = (M, N, [x, y])
            in normalized form [0, 1]
    """
    return landmarks / tf.cast([width, height], tf.float32)

def denormalize_landmarks(landmarks, height, width):
    """Denormalizing landmarks.
    inputs:
        landmarks = (M, N, [x, y])
            in normalized form [0, 1]
        height = image height
        width = image width

    outputs:
        denormalized_landmarks = (M, N, [x, y])
    """
    return tf.round(landmarks * tf.cast([width, height], tf.float32))
