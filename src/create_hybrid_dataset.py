import tensorflow as tf

from utils import load_tfrecord, write_tfrecord

# Define your default dataset paths
real_dataset_path = '../processed_datasets/real.tfrecord'
synthetic_dataset_path = 'minority_synthetic_data1'
new_hybrid_path = '../processed_datasets/hybrid_large_balanced1.tfrecord'

# batching is done because image_dataset_from_directory batches automatically
# and this ensures proper shapes also for later use
batch_size = 32
# load real dataset
real_dataset = load_tfrecord(real_dataset_path)

real_dataset = real_dataset.batch(batch_size)

for x, y in real_dataset.take(1):
    print(x.shape, y.shape)

# load synthetic dataset
synthetic_dataset = tf.keras.utils.image_dataset_from_directory(
        synthetic_dataset_path, label_mode='int', batch_size=batch_size)

synthetic_dataset = synthetic_dataset.map(
        lambda x, y: (tf.cast(x, tf.uint8), tf.cast(y, tf.int64)))

for x, y in real_dataset.take(1):
    print(x.shape, y.shape)

dataset = real_dataset.concatenate(synthetic_dataset)

# merge them and write them to another file
write_tfrecord(dataset, new_hybrid_path, scale_image_back=False)
