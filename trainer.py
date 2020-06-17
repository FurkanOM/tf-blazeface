import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
import augmentation
from ssd_loss import CustomLoss
from utils import bbox_utils, data_utils, io_utils, train_utils, drawing_utils, landmark_utils
import blazeface
import random

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 32
epochs = 150
load_weights = False
hyper_params = train_utils.get_hyper_params()

train_split = "train[:80%]"
val_split = "train[80%:]"
train_data, info = data_utils.get_dataset("the300w_lp", train_split)
val_data, _ = data_utils.get_dataset("the300w_lp", val_split)
train_total_items = data_utils.get_total_item_size(info, train_split)
val_total_items = data_utils.get_total_item_size(info, val_split)
#
img_size = hyper_params["img_size"]

train_data = train_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, augmentation.apply))
val_data = val_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))
#
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()
train_data = train_data.shuffle(batch_size*12).padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
val_data = val_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
#
model = blazeface.get_model(hyper_params)
custom_losses = CustomLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])
model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=[custom_losses.loc_loss_fn, custom_losses.conf_loss_fn])
blazeface.init_model(model)
#
model_path = io_utils.get_model_path()
if load_weights:
    model.load_weights(model_path)
log_path = io_utils.get_log_path("blazeface/")
# We calculate prior boxes for one time and use it for all operations because of the all images are the same sizes
prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
#
train_feed = train_utils.generator(train_data, prior_boxes, hyper_params)
val_feed = train_utils.generator(val_data, prior_boxes, hyper_params)

checkpoint_callback = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=log_path)
learning_rate_callback = LearningRateScheduler(train_utils.scheduler, verbose=0)

step_size_train = train_utils.get_step_size(train_total_items, batch_size)
step_size_val = train_utils.get_step_size(val_total_items, batch_size)
model.fit(train_feed,
          steps_per_epoch=step_size_train,
          validation_data=val_feed,
          validation_steps=step_size_val,
          epochs=epochs,
          callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback])
