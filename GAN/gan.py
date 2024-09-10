import os
from datetime import datetime
import argparse
import ast

import tensorflow as tf
import tensorflow.keras as keras

import model
from src.utils import load_tfrecord, get_latest_checkpoint

# Define the default values
batch_size = 256 # set this as high as possible as GANs profit from larger batch sizes
num_channels = 3
num_classes = 55
image_size = 224
latent_dim = 128
image_dir = "../images"
checkpoint_dir = '../ckpt'
EPOCHS = 10
# datasets_to_use = ['../processed_datasets/train1.tfrecord', '../processed_datasets/validation1.tfrecord']
datasets_to_use = ['../processed_datasets/train_balanced_undersampled.tfrecord']

# Set up the argument parser
parser = argparse.ArgumentParser(description="Script for training a model with specified parameters.")
parser.add_argument('--batch_size', type=int, default=batch_size,
                    help='Batch size for training (default: {})'.format(batch_size))
parser.add_argument('--num_channels', type=int, default=num_channels,
                    help='Number of channels in the images (default: {})'.format(num_channels))
parser.add_argument('--num_classes', type=int, default=num_classes,
                    help='Number of classes in the dataset (default: {})'.format(num_classes))
parser.add_argument('--image_size', type=int, default=image_size,
                    help='Size of the images (default: {})'.format(image_size))
parser.add_argument('--latent_dim', type=int, default=latent_dim,
                    help='Latent dimension for the model (default: {})'.format(latent_dim))
parser.add_argument('--image_dir', type=str, default=image_dir,
                    help='Directory of images (default: {})'.format(image_dir))
parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir,
                    help='Directory to save model checkpoints/weights (default: {})'.format(checkpoint_dir))
parser.add_argument('--epochs', type=int, default=EPOCHS,
                    help='Number of training epochs (default: {})'.format(EPOCHS))
parser.add_argument('--datasets_to_use', type=str, default=str(datasets_to_use),
                    help='List of datasets to use for training. These should be paths to tfrecord files (default: {}).'.format(datasets_to_use))

# Parse the command-line arguments
args = parser.parse_args()

# Convert the datasets_to_use string back to a list
datasets_to_use = ast.literal_eval(args.datasets_to_use)

print("Batch Size:", args.batch_size)
print("Number of Channels:", args.num_channels)
print("Number of Classes:", args.num_classes)
print("Image Size:", args.image_size)
print("Latent Dimension:", args.latent_dim)
print("Image Directory:", args.image_dir)
print("Checkpoint Directory:", args.checkpoint_dir)
print("Number of Epochs:", args.epochs)
print("Datasets to Use:", datasets_to_use)


def prepare_dataset(data):
    """
    Input Pipeline which prepares the dataset for further processing
    :param data: the dataset
    :return: preprocessed dataset
    """

    data = data.map(lambda image, target: (image,
                                           tf.keras.utils.to_categorical(target,
                                                                         num_classes=num_classes)))
    data = data.map(
            lambda image, target: (tf.cast(image, tf.float32) / 255.0, target))
    # data = data.cache()
    data = data.shuffle(1000)
    data = data.batch(batch_size)
    data = data.prefetch(20)
    return data


dataset = load_tfrecord(datasets_to_use.pop(0))

# concatenate the datasets to train the GAN on all the data
while len(datasets_to_use)>0:
    concatenate_dataset = load_tfrecord(datasets_to_use.pop(0))
    dataset = dataset.concatenate(concatenate_dataset)


dataset = dataset.apply(prepare_dataset)
dataset = dataset.take(2)

# [TODO] for the first stage undersample all classes to the smallest number of samples of a class to train on a balanced dataset

# After loading the dataset and applying the above operations
for images, labels in dataset.take(1):  # Take just one batch to check the shape
    print("Batch Size (Images):",
          images.shape)  # Shape will be (batch_size, height, width, channels)
    print("Batch Size (Labels):", labels.shape)  # Shape will be (batch_size,)

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

# Create the discriminator.
discriminator = model.get_discriminator_model(image_size,
                                              discriminator_in_channels)
# Create the generator.
generator = model.get_generator_model(generator_in_channels)

print("discriminator", discriminator.summary())
print("generator", generator.summary())

# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5,
                                            beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002,
                                                beta_1=0.5, beta_2=0.9)


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
generator_loss = keras.losses.BinaryCrossentropy(from_logits=True)

# Instantiate the customer `GANMonitor` Keras callback.
cbk = model.GANMonitor(image_dir=image_dir, num_classes=num_classes, num_img=3,
                       latent_dim=latent_dim)

# Get the wgan model
wgan = model.ConditionalWGAN(discriminator=discriminator, generator=generator,
                             latent_dim=latent_dim, image_size=image_size,
                             num_classes=num_classes,
                             discriminator_extra_steps=3, )

# Compile the wgan model
wgan.compile(d_optimizer=discriminator_optimizer,
             g_optimizer=generator_optimizer, g_loss_fn=generator_loss,
             d_loss_fn=discriminator_loss, )

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

checkpoint_filepath = os.path.join(checkpoint_dir,
                                   'checkpoint-{epoch:02d}.weights.h5')

# Get the latest checkpoint file path
latest_checkpoint_path = get_latest_checkpoint(checkpoint_dir)

if latest_checkpoint_path:
    # Load the model from the latest checkpoint
    wgan.load_weights(latest_checkpoint_path)
    print(f"Loaded model weights from {latest_checkpoint_path}")
else:
    print("No checkpoint found.")

print(wgan.metrics_names)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, monitor='generator_loss',
        save_weights_only=True, mode='min', verbose=1, save_best_only=False)

print("Fitting the GAN")
wgan.fit(dataset, batch_size=batch_size, epochs=EPOCHS,
         callbacks=[tensorboard_callback, model_checkpoint_callback, cbk])
