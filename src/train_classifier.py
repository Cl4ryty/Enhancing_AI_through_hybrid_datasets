import tensorflow as tf

from src.pipeline import finetuning_pipeline
from src.utils import load_tfrecord

import argparse

# Define the default values
batch_size = 32
top_layer_epochs = 30
end_to_end_epochs = 5
train_dataset_path = '../train.tfrecord'
validation_dataset_path = '../validation.tfrecord'
test_dataset_path = '../test.tfrecord'

# Set up the argument parser
parser = argparse.ArgumentParser(description="Script for finetuning a model with specified parameters.")
parser.add_argument('--batch_size', type=int, default=batch_size,
                    help='Batch size for training (default: {})'.format(batch_size))
parser.add_argument('--top_layer_epochs', type=int, default=top_layer_epochs,
                    help='Number of epochs for the top layer training (default: {})'.format(top_layer_epochs))
parser.add_argument('--end_to_end_epochs', type=int, default=end_to_end_epochs,
                    help='Number of epochs for end-to-end training (default: {})'.format(end_to_end_epochs))
parser.add_argument('--train_dataset_path', type=str, default=train_dataset_path,
                    help='Path to the training dataset (default: {})'.format(train_dataset_path))
parser.add_argument('--validation_dataset_path', type=str, default=validation_dataset_path,
                    help='Path to the validation dataset (default: {})'.format(validation_dataset_path))
parser.add_argument('--test_dataset_path', type=str, default=test_dataset_path,
                    help='Path to the test dataset (default: {})'.format(test_dataset_path))

# Parse the command-line arguments
args = parser.parse_args()

print("Batch Size:", args.batch_size)
print("Top Layer Epochs:", args.top_layer_epochs)
print("End-to-End Epochs:", args.end_to_end_epochs)
print("Training Dataset Path:", args.train_dataset_path)
print("Validation Dataset Path:", args.validation_dataset_path)
print("Test Dataset Path:", args.test_dataset_path)


train_dataset = load_tfrecord(train_dataset_path)
validation_dataset = load_tfrecord(validation_dataset_path)
test_dataset = load_tfrecord(test_dataset_path)

# Shuffle, Batch, and Prefetch
train_dataset = train_dataset.shuffle(buffer_size=1000)  # Shuffle the dataset
train_dataset = train_dataset.batch(batch_size)  # Batch the dataset
train_dataset = train_dataset.prefetch(
    tf.data.experimental.AUTOTUNE)  # Prefetch for performance

validation_dataset = validation_dataset.shuffle(
    buffer_size=1000)  # Shuffle the dataset
validation_dataset = validation_dataset.batch(batch_size)  # Batch the dataset
validation_dataset = validation_dataset.prefetch(
    tf.data.experimental.AUTOTUNE)  # Prefetch for performance

test_dataset = test_dataset.shuffle(buffer_size=1000)  # Shuffle the dataset
test_dataset = test_dataset.batch(batch_size)  # Batch the dataset
test_dataset = test_dataset.prefetch(
    tf.data.experimental.AUTOTUNE)  # Prefetch for performance

# Check the shape of images and labels in one batch
for images, labels in train_dataset.take(
        1):
    print("Batch Size (Images):",
          images.shape)  # (batch_size, height, width, channels)
    print("Batch Size (Labels):", labels.shape)  # (batch_size,)

base_model = tf.keras.applications.MobileNet(include_top=False,
        weights='imagenet', input_shape=(224, 224, 3), input_tensor=None,
        pooling=None, )

finetuning_pipeline(base_model, train_dataset, validation_dataset, test_dataset,
                    number_of_classes=55, top_layer_epochs=top_layer_epochs, end_to_end_epochs=end_to_end_epochs)
