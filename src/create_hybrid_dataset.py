import tensorflow as tf
import argparse
from utils import load_tfrecord, get_latest_checkpoint, write_tfrecord


# Define your default dataset paths
real_dataset_path = '../processed_datasets/test1.tfrecord'
synthetic_dataset_path = '../GAN/images_generated'
new_hybrid_path = '../processed_datasets/hybrid_large_balanced.tfrecord'

# Set up the argument parser
parser = argparse.ArgumentParser(description="Script for processing creating a hybrid dataset from a tfrecord of a real dataset and a synthetic dataset that exists as images in a directory.")
parser.add_argument('--real_dataset_path', type=str, default=real_dataset_path,
                    help='Path to the real dataset. This should be a tfrecord  (default: {}).'.format(real_dataset_path))
parser.add_argument('--synthetic_dataset_path', type=str, default=synthetic_dataset_path,
                    help='Path to the synthetic dataset. This should be a directory containing subdirectories of the classes  (default: {}).'.format(synthetic_dataset_path))
parser.add_argument('--new_hybrid_path', type=str, default=new_hybrid_path,
                    help='Path to save the new hybrid dataset. Should include the filename with .tfrecord extension (default: {}).'.format(new_hybrid_path))

# Parse the command-line arguments
args = parser.parse_args()


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
        synthetic_dataset_path,
        label_mode='int',
        batch_size=batch_size)

synthetic_dataset = synthetic_dataset.map(lambda x, y: (tf.cast(x, tf.uint8), tf.cast(y, tf.int64)))

for x, y in real_dataset.take(1):
    print(x.shape, y.shape)


dataset = real_dataset.concatenate(synthetic_dataset)

# merge them and write them to another file
write_tfrecord(dataset, new_hybrid_path, scale_image_back=False)