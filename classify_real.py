import tensorflow as tf

from pipeline import finetuning_pipeline


def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def load_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    parsed_dataset = parsed_dataset.map(lambda x: (tf.image.decode_jpeg(x['image']), x['label']))  # Decode the image
    return parsed_dataset


train_dataset = load_tfrecord('train.tfrecord')
validation_dataset = load_tfrecord('validation.tfrecord')
test_dataset = load_tfrecord('test.tfrecord')

# Shuffle, Batch, and Prefetch
batch_size = 32  # Set your desired batch size
train_dataset = train_dataset.shuffle(buffer_size=1000)  # Shuffle the dataset
train_dataset = train_dataset.batch(batch_size)  # Batch the dataset
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch for performance

validation_dataset = validation_dataset.shuffle(buffer_size=1000)  # Shuffle the dataset
validation_dataset = validation_dataset.batch(batch_size)  # Batch the dataset
validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch for performance

test_dataset = test_dataset.shuffle(buffer_size=1000)  # Shuffle the dataset
test_dataset = test_dataset.batch(batch_size)  # Batch the dataset
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch for performance




# After loading the dataset and applying the above operations
for images, labels in train_dataset.take(1):  # Take just one batch to check the shape
    print("Batch Size (Images):", images.shape)  # Shape will be (batch_size, height, width, channels)
    print("Batch Size (Labels):", labels.shape)  # Shape will be (batch_size,)

base_model = tf.keras.applications.MobileNet(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    input_tensor=None,
    pooling=None,
)

finetuning_pipeline(base_model, train_dataset, validation_dataset, test_dataset, number_of_classes=55)
