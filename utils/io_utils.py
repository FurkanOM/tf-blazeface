import os
import argparse
import tensorflow as tf
from datetime import datetime

def get_log_path(custom_prefix=""):
    """Generating log path for tensorboard.
    inputs:
        custom_prefix = any custom string for log folder name

    outputs:
        log_path = tensorboard log path, for example: "logs/{custom_prefix}{date}"
    """
    return "logs/{}{}".format(custom_prefix, datetime.now().strftime("%Y%m%d-%H%M%S"))

def get_model_path():
    """Generating model path for save/load model weights.

    outputs:
        model_path = os model path, for example: "models/blazeface_model_weights.h5"
    """
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "blazeface_model_weights.h5")
    return model_path

def handle_args():
    """Handling of command line arguments using argparse library.

    outputs:
        args = parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs Implementation")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    args = parser.parse_args()
    return args

def handle_gpu_compatibility():
    """Handling of GPU issues for cuDNN initialize error and memory issues."""
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)
