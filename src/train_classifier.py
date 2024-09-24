import os
import re
from datetime import datetime
import time

import tensorflow as tf

from pipeline import finetuning_pipeline
from utils import load_tfrecord, get_latest_checkpoint

import argparse

# Define the default values
batch_size = 256
top_layer_epochs = 80
end_to_end_epochs = 30
train_dataset_path = '/home/student/h/hakoester/share/processed_datasets/train1.tfrecord'
validation_dataset_path = '/home/student/h/hakoester/share/processed_datasets/validation1.tfrecord'
test_dataset_path = '/home/student/h/hakoester/share/processed_datasets/test1.tfrecord'
number_of_classes = 55

log_name = ''

checkpoint_dir = '/home/student/h/hakoester/share/checkpoints/classifier_real_full/'

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
parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir)

parser.add_argument('--log_name', type=str, default=log_name)

# Parse the command-line arguments
args = parser.parse_args()

# Now set variables with the same names
batch_size = args.batch_size
top_layer_epochs = args.top_layer_epochs
end_to_end_epochs = args.end_to_end_epochs
train_dataset_path = args.train_dataset_path
validation_dataset_path = args.validation_dataset_path
test_dataset_path = args.test_dataset_path
log_name = args.log_name
checkpoint_dir = args.checkpoint_dir


os.makedirs(checkpoint_dir, exist_ok=True)

# Print the variables to check their values
print("Batch Size:", batch_size)
print("Top Layer Epochs:", top_layer_epochs)
print("End-to-End Epochs:", end_to_end_epochs)
print("Training Dataset Path:", train_dataset_path)
print("Validation Dataset Path:", validation_dataset_path)
print("Test Dataset Path:", test_dataset_path)
print("Log Name:", log_name)
print("Checkpoint Dir:", checkpoint_dir)

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
#
# train_dataset = train_dataset.take(10)
# validation_dataset = validation_dataset.take(10)
# test_dataset = test_dataset.take(10)

# Check the shape of images and labels in one batch
for images, labels in train_dataset.take(
        1):
    print("Batch Size (Images):",
          images.shape)  # (batch_size, height, width, channels)
    print("Batch Size (Labels):", labels.shape)  # (batch_size,)

base_model = tf.keras.applications.MobileNet(include_top=False,
        weights='imagenet', input_shape=(224, 224, 3), input_tensor=None,
        pooling=None, )


# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = tf.keras.Input(shape=(224, 224, 3))
# preprocess inputs (scaling to expected range of -1 to 1)
x = tf.keras.applications.mobilenet.preprocess_input(inputs)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = tf.keras.layers.Dense(number_of_classes)(
    x)  # units for classifying
model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True), metrics=['accuracy'], )

log_dir = os.path.join("logs/fit", log_name, datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

# Set up the ModelCheckpoint callback to save weights
checkpoint_filepath = os.path.join(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,  # Save only the weights
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',  # Save the model with the maximum validation accuracy
        save_best_only=True  # Save the best model
)

# Regular expression to match the checkpoint pattern and extract the epoch number
pattern = re.compile(r"weights\.(\d{2})-(\d+\.\d+)\.weights\.h5")

# Get the latest checkpoint file path
latest_checkpoint_path, start_epoch = get_latest_checkpoint(checkpoint_dir, pattern)

# # Calculate the adjusted epochs
# if start_epoch <= top_layer_epochs:
#     top_layer_epochs -= start_epoch
# else:
#     top_layer_epochs = 0  # No epochs left for top layer training if started beyond that
#     start_epoch -= top_layer_epochs
#
# if start_epoch <= end_to_end_epochs:
#     end_to_end_epochs -= start_epoch
# else:
#     end_to_end_epochs = 0  # No epochs left for end-to-end training if started beyond that


if latest_checkpoint_path:
    print("latest checkpoint path", latest_checkpoint_path)
    # Load the model from the latest checkpoint
    model.load_weights(latest_checkpoint_path)
    print(f"Loaded model weights from {latest_checkpoint_path}")
else:
    print("No checkpoint found.")

print("Fitting the top layer of the model")
start_time = time.time()

model.fit(train_dataset, epochs=top_layer_epochs, validation_data=validation_dataset,
         callbacks=[tensorboard_callback, checkpoint_callback])
print("Time taken: %.2fs" % (time.time() - start_time))

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True), metrics=['accuracy'], )

print("Fitting the end-to-end model")
start_time = time.time()
model.fit(train_dataset, epochs=end_to_end_epochs+top_layer_epochs, initial_epoch=top_layer_epochs, validation_data=validation_dataset,
          callbacks=[tensorboard_callback, checkpoint_callback])
print("Time taken: %.2fs" % (time.time() - start_time))

print("Test dataset evaluation")
result = model.evaluate(test_dataset)
print("test results:", dict(zip(model.metrics_names, result)))
