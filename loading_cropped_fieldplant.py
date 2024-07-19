import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt

from pipeline import finetuning_pipeline

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64

def read_tfrecord(record):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
    )
    record = tf.io.parse_single_example(record, tfrecord_format)
    image = tf.image.decode_image(record["image"], channels=3, expand_animations = False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, tf.constant([224, 224]))
    label = tf.cast(record["label"], tf.int32)-1
    return image, label


def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord), num_parallel_calls=AUTOTUNE
    )
    return dataset


# Path to your TFRecord file
tfrecord_file = "cropped_dataset.tfrecords"
dataset = load_dataset(tfrecord_file)


dataset_size = tf.data.experimental.cardinality(dataset).numpy()

print("Dataset size:", dataset_size)
dataset_size=8629
dataset = dataset.shuffle(buffer_size=2048)

# Enumerate dataset
enumerated_dataset = dataset.enumerate()
print("Dataset size:", dataset_size)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

# Filter datasets based on indices for train, validation, and test splits
train_dataset = enumerated_dataset.filter(lambda i, _: i < train_size).map(lambda i, x: x)
val_dataset = enumerated_dataset.filter(lambda i, _: train_size <= i < train_size + val_size).map(lambda i, x: x)
test_dataset = enumerated_dataset.filter(lambda i, _: i >= train_size + val_size).map(lambda i, x: x)


# Verify the sizes
print("Train size:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Validation size:", tf.data.experimental.cardinality(val_dataset).numpy())
print("Test size:", tf.data.experimental.cardinality(test_dataset).numpy())

print("Dataset size:", dataset_size)

# Calculate the sizes for each subset
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

print("train size", train_size, "val size", val_size, "test size", test_size)

# Split the dataset
train_ds = dataset.take(train_size)
remaining_ds = dataset.skip(train_size)
validation_ds = remaining_ds.take(val_size)
test_ds = remaining_ds.skip(val_size)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


image_batch, label_batch = next(iter(train_ds))

print(image_batch.shape)
print(label_batch.shape)

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis("off")

    plt.show()

show_batch(image_batch.numpy(), label_batch.numpy())


base_model = tf.keras.applications.MobileNet(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    input_tensor=None,
    pooling=None,
)

finetuning_pipeline(base_model, train_ds, validation_ds, test_ds, number_of_classes=27)

