import tensorflow as tf

def normalize_landmarks(landmarks, height, width):
    """Normalizing landmarks.
    inputs:
        bboxes = (M x N, [x, y])
        height = image height
        width = image width

    outputs:
        normalized_bboxes = (M x N, [x, y])
            in normalized form [0, 1]
    """
    return landmarks / tf.cast([width, height], tf.float32)

def denormalize_landmarks(landmarks, height, width):
    """Denormalizing landmarks.
    inputs:
        landmarks = (M x N, [x, y])
            in normalized form [0, 1]
        height = image height
        width = image width

    outputs:
        denormalized_landmarks = (M x N, [x, y])
    """
    return tf.round(landmarks * tf.cast([width, height], tf.float32))
