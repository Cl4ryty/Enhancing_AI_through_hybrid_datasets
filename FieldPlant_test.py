from src.pipeline import finetuning_pipeline

import tensorflow as tf

# Path to your TFRecord file
tfrecord_file = "datasets/FieldPlant/train/leaves.tfrecord"

# Read the TFRecord file
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

# Define the feature description
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64)
}

# Define the parsing function
def _parse_function(proto):
    return tf.io.parse_single_example(proto, feature_description)

# Parse the dataset
parsed_dataset = raw_dataset.map(_parse_function)

# Decode the image and process the labels
def decode_image_and_label(parsed_record):
    image = tf.io.decode_image(parsed_record['image/encoded'])
    label = tf.sparse.to_dense(parsed_record['image/object/class/label'])

    # Convert sparse tensors to dense tensors
    xmin = tf.sparse.to_dense(parsed_record['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(parsed_record['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(parsed_record['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(parsed_record['image/object/bbox/ymax'])

    # Stack them together to form bounding boxes
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

    return image, (label, bboxes)

# Let's assume `parsed_dataset` has been defined previously
dataset = parsed_dataset.map(decode_image_and_label)

# Batch and prefetch the dataset
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

# Calculate the dataset size
dataset_size = tf.data.experimental.cardinality(dataset).numpy()

# Calculate the sizes for each subset
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the dataset
train_ds = dataset.take(train_size)
remaining_ds = dataset.skip(train_size)
validation_ds = remaining_ds.take(val_size)
test_ds = remaining_ds.skip(val_size)


base_model = tf.keras.applications.MobileNet(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    input_tensor=None,
    pooling=None,
)

finetuning_pipeline(base_model, train_ds, validation_ds, test_ds, number_of_classes=27)

