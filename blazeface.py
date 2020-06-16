import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, DepthwiseConv2D, Conv2D, MaxPool2D, Add, Activation

class HeadWrapper(Layer):
    """Merging all feature maps for detections.
    inputs:
        conv4_3 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv4_3 shape => (38 x 38 x 4) = 5776
        conv7 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv7 shape => (19 x 19 x 6) = 2166
        conv8_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv8_2 shape => (10 x 10 x 6) = 600
        conv9_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv9_2 shape => (5 x 5 x 6) = 150
        conv10_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv10_2 shape => (3 x 3 x 4) = 36
        conv11_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv11_2 shape => (1 x 1 x 4) = 4
                                           Total = 8732 default box

    outputs:
        merged_head = (batch_size, total_bboxes, last_dimension)
    """

    def __init__(self, last_dimension, **kwargs):
        super(HeadWrapper, self).__init__(**kwargs)
        self.last_dimension = last_dimension

    def get_config(self):
        config = super(HeadWrapper, self).get_config()
        config.update({"last_dimension": self.last_dimension})
        return config

    def call(self, inputs):
        last_dimension = self.last_dimension
        batch_size = tf.shape(inputs[0])[0]
        outputs = []
        for conv_layer in inputs:
            outputs.append(tf.reshape(conv_layer, (batch_size, -1, last_dimension)))
        #
        return tf.concat(outputs, axis=1)

def blaze_block(input, filters, stride=1):
    y = input
    x = DepthwiseConv2D((5,5), strides=stride, padding="same")(input)
    x = Conv2D(filters, (1,1), padding="same")(x)
    if stride == 2:
        y = MaxPool2D((2,2))(y)
        y = Conv2D(filters, (1,1), padding="same")(y)
    output = Add()([x, y])
    return Activation("relu")(output)

def double_blaze_block(input, filters, stride=1):
    y = input
    x = DepthwiseConv2D((5,5), strides=stride, padding="same")(input)
    x = Conv2D(filters[0], (1,1), padding="same")(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((5,5), padding="same")(x)
    x = Conv2D(filters[1], (1,1), padding="same")(x)
    if stride == 2:
        y = MaxPool2D((2,2))(y)
        y = Conv2D(filters[1], (1,1), padding="same")(y)
    output = Add()([x, y])
    return Activation("relu")(output)

def get_model(hyper_params):
    detections_per_layer = hyper_params["detections_per_layer"]
    img_size = hyper_params["img_size"]
    total_reg_points = hyper_params["total_landmarks"] * 2 + 4
    #
    input = Input(shape=(None, None, 3))
    # First conv layer
    first_conv = Conv2D(24, (5,5), strides=2, padding="same", activation="relu")(input)
    # First blaze block
    single_1 = blaze_block(first_conv, 24)
    # Second blaze block
    single_2 = blaze_block(single_1, 24)
    # Third blaze block
    single_3 = blaze_block(single_2, 48, 2)
    # Fourth blaze block
    single_4 = blaze_block(single_3, 48)
    # Fifth blaze block
    single_5 = blaze_block(single_4, 48)
    # First double blaze block
    double_1 = double_blaze_block(single_5, [24, 96], 2)
    # Second double blaze block
    double_2 = double_blaze_block(double_1, [24, 96])
    # Third double blaze block
    double_3 = double_blaze_block(double_2, [24, 96])
    # Fourth double blaze block
    double_4 = double_blaze_block(double_3, [24, 96], 2)
    # Fifth double blaze block
    double_5 = double_blaze_block(double_4, [24, 96])
    # Sixth double blaze block
    double_6 = double_blaze_block(double_5, [24, 96])
    #
    double_3_labels = Conv2D(detections_per_layer[0], (3, 3), padding="same")(double_3)
    double_6_labels = Conv2D(detections_per_layer[1], (3, 3), padding="same")(double_6)
    #
    double_3_boxes = Conv2D(detections_per_layer[0] * total_reg_points, (3, 3), padding="same")(double_3)
    double_6_boxes = Conv2D(detections_per_layer[1] * total_reg_points, (3, 3), padding="same")(double_6)
    #
    pred_labels = HeadWrapper(1, name="conf_head")([double_3_labels, double_6_labels])
    pred_labels = Activation("sigmoid", name="conf")(pred_labels)
    pred_deltas = HeadWrapper(total_reg_points, name="loc")([double_3_boxes, double_6_boxes])
    #
    return Model(inputs=input, outputs=[pred_deltas, pred_labels])

def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model

    """
    model(tf.random.uniform((1, 512, 512, 3)))
