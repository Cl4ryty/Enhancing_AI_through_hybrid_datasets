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

dataset_size=8629
dataset = dataset.shuffle(buffer_size=2048)

# Enumerate dataset
enumerated_dataset = dataset.enumerate()
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

# Filter datasets based on indices for train, validation, and test splits
train_ds = enumerated_dataset.filter(lambda i, _: i < train_size).map(lambda i, x: x)
val_ds = enumerated_dataset.filter(lambda i, _: train_size <= i < train_size + val_size).map(lambda i, x: x)
test_ds = enumerated_dataset.filter(lambda i, _: i >= train_size + val_size).map(lambda i, x: x)


# Verify the sizes
print("Dataset size:", dataset_size)
print("train size", train_size, "val size", val_size, "test size", test_size)

# # Split the dataset
# train_ds = dataset.take(train_size)
# remaining_ds = dataset.skip(train_size)
# validation_ds = remaining_ds.take(val_size)
# test_ds = remaining_ds.skip(val_size)
#
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


image_batch, label_batch = next(iter(train_ds))

print(image_batch.shape)
print(label_batch.shape)
#
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis("off")

    plt.show()
#
show_batch(image_batch.numpy(), label_batch.numpy())


# load nzdl dataset
train_data = tf.keras.utils.image_dataset_from_directory('datasets/NZDL/train',
                                                       seed=123, batch_size=BATCH_SIZE)

validation_data = tf.keras.utils.image_dataset_from_directory(
        'datasets/NZDL/valid', seed=123, batch_size=BATCH_SIZE)

test_data = tf.keras.utils.image_dataset_from_directory('datasets/NZDL/test',
                                                      seed=123, batch_size=BATCH_SIZE)



# change labels for nzdl
# Define the mapping function
def map_function(features, labels):
    labels += 27
    mask = tf.equal(labels, 26+27)  # Returns a boolean tensor

    # Replace specific labels with new_value
    modified_labels = tf.where(mask, 24, labels)

    return features, modified_labels

# Apply the label change to dataset2
train_data = train_data.map(map_function)
test_data = test_data.map(map_function)
validation_data = validation_data.map(map_function)

# reformat the image
train_data = train_data.map(lambda image, label: (tf.cast(image, tf.float32) / 255.0, label))
train_data = train_data.map(lambda image, label: (tf.image.resize(image, (224,224)), label))
test_data = test_data.map(lambda image, label: (tf.cast(image, tf.float32) / 255.0, label))
test_data = test_data.map(lambda image, label: (tf.image.resize(image, (224,224)), label))
validation_data = validation_data.map(lambda image, label: (tf.cast(image, tf.float32) / 255.0, label))
validation_data = validation_data.map(lambda image, label: (tf.image.resize(image, (224,224)), label))

print("resized images")
image_batch, label_batch = next(iter(train_data))

print(image_batch.shape)
print(label_batch.shape)
show_batch(image_batch.numpy(), label_batch.numpy())

# merge into one dataset
# resize
combined_train = train_data.concatenate(train_ds)
combined_test = test_data.concatenate(test_ds)
combined_validation = validation_data.concatenate(validation_ds)
print("combined")

image_batch, label_batch = next(iter(train_data))

print(image_batch.shape)
print(label_batch.shape)
show_batch(image_batch.numpy(), label_batch.numpy())




# save the combined datasets so that they can be directly loaded without having to do the preprocessing again
def write_tfrecord(dataset, filename):
    writer = tf.io.TFRecordWriter(filename)
    i = 0

    for batch_features, batch_labels in dataset:
        # Loop through each image in the batch
        for features, label in zip(batch_features, batch_labels):
            # Scale the float image to the range [0, 255] and convert to uint8
            scaled_image = tf.cast(features * 255, tf.uint8)
            if i < 5:
                plt.imshow(scaled_image)
                plt.show()
                i += 1
            # Encode the image as JPEG
            encoded_image = tf.io.encode_jpeg(scaled_image)


            # Create a tf.train.Example
            example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[encoded_image.numpy()])),
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label.numpy()]))}))
            writer.write(example.SerializeToString())

    writer.close()

write_tfrecord(combined_train, 'train.tfrecord')
write_tfrecord(combined_validation, 'validation.tfrecord')
write_tfrecord(combined_test, 'test.tfrecord')


