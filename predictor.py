import tensorflow as tf
from utils import bbox_utils, data_utils, drawing_utils, io_utils, train_utils, landmark_utils
import blazeface

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 1
use_custom_images = True
custom_image_path = "data/images/"
#
hyper_params = train_utils.get_hyper_params()
#
img_size = hyper_params["img_size"]

if use_custom_images:
    test_data = data_utils.get_image_data_from_folder(custom_image_path, img_size, img_size)
else:
    files = data_utils.get_files_from_folder("data/modified")
    dataset_types = data_utils.get_dataset_types()
    dataset_shapes = data_utils.get_dataset_shapes()
    test_data = tf.data.Dataset.from_generator(lambda: data_utils.dataset_generator(files),
                                                dataset_types, dataset_shapes)
    test_data = test_data.map(lambda a,b,c : data_utils.preprocessing((a,b,c), img_size, img_size))
    padding_values = data_utils.get_batch_paddings()
    test_data = test_data.padded_batch(batch_size, padded_shapes=dataset_shapes, padding_values=padding_values)

model = blazeface.get_model(hyper_params)
model_path = io_utils.get_model_path()
model.load_weights(model_path)

prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])

variances = hyper_params["variances"]
landmark_pair_count = hyper_params["landmark_pair_count"]
landmark_variances = landmark_pair_count * variances[0:2]
variances += landmark_variances

for image_data in test_data:
    img, _, _ = image_data
    pred_deltas, pred_scores = model.predict_on_batch(img)
    pred_deltas *= variances
    #
    pred_bboxes_and_landmarks = bbox_utils.get_bboxes_and_landmarks_from_deltas(prior_boxes, pred_deltas)
    pred_bboxes_and_landmarks = tf.clip_by_value(pred_bboxes_and_landmarks, 0, 1)
    #
    pred_scores = tf.cast(pred_scores, tf.float32)
    #
    weighted_suppressed_data = bbox_utils.weighted_suppression(pred_scores[0], pred_bboxes_and_landmarks[0])
    #
    weighted_bboxes = weighted_suppressed_data[..., 0:4]
    weighted_landmarks = weighted_suppressed_data[..., 4:]
    #
    denormalized_bboxes = bbox_utils.denormalize_bboxes(weighted_bboxes, img_size, img_size)
    weighted_landmarks = tf.reshape(weighted_landmarks, (1, -1, landmark_pair_count, 2))
    denormalized_landmarks = landmark_utils.denormalize_landmarks(weighted_landmarks, img_size, img_size)
    drawing_utils.draw_bboxes_with_landmarks(img[0], denormalized_bboxes, denormalized_landmarks)
