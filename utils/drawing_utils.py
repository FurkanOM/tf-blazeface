import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_bboxes(imgs, bboxes):
    """Drawing bounding boxes on given images.
    inputs:
        imgs = (batch_size, height, width, channels)
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    for img_with_bb in imgs_with_bb:
        plt.imshow(img_with_bb)
        plt.show()

def draw_bboxes_with_landmarks(img, bboxes, landmarks):
    """Drawing bounding boxes and landmarks on given image.
    inputs:
        img = (height, width, channels)
        bboxes = (M x N, [y1, x1, y2, x2])
        landmarks = (M x N, [x, y])
    """
    image = tf.keras.preprocessing.image.array_to_img(img)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    color = (255, 0, 0, 255)
    for index, bbox in enumerate(bboxes):
        y1, x1, y2, x2 = tf.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=color, width=1)
    for index, landmark in enumerate(landmarks):
        if tf.reduce_max(landmark) <= 0:
            continue
        rects = tf.concat([landmark - 1, landmark + 1], -1)
        for rect in rects:
            draw.ellipse(rect, fill=color)
    plt.figure()
    plt.imshow(image)
    plt.show()
