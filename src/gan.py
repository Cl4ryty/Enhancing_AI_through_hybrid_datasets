import os
import re
from datetime import datetime
import argparse
import ast

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['XLA_FLAGS'] = '/appl/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/cuda-11.8.0-x32erfzo6xl2qgbp5enezl53wwiingmt/nvvm'
print ("Working directory:" , os.getcwd()) 

import tensorflow as tf
import tensorflow.keras as keras

import model
from utils import load_tfrecord, get_latest_checkpoint

print("tf gpus:", tf.config.list_physical_devices('GPU'))

# Define the default values
batch_size = 256 # set this as high as possible as GANs profit from larger batch sizes
num_channels = 3
num_classes = 55
image_size = 224
latent_dim = 128
image_dir = "/home/student/h/hakoester/share/cgan_s1_images"
checkpoint_dir = '/home/student/h/hakoester/share/checkpoints/cgan_s1'

# image_dir = "../s1_gpu_images"
# checkpoint_dir = '../checkpoints/gan_s1'
EPOCHS = 4000
start_epoch = 0
# datasets_to_use = ['../processed_datasets/train1.tfrecord', '../processed_datasets/validation1.tfrecord']
datasets_to_use = ['/home/student/h/hakoester/share/train_balanced_undersampled.tfrecord']
# datasets_to_use = ['/home/hannah/Documents/A0_uni/master/S2/EnhancingAI/project_test/project_test/processed_datasets/train_balanced_undersampled.tfrecord']

save_frequency = 25

discriminator_extra_steps = 1

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

    data = data.map(lambda image, target: (image, tf.one_hot(target, depth=num_classes)))
    data = data.map(
            lambda image, target: (tf.cast(image, tf.float32) / 255.0, target))
    # data = data.cache()
    data = data.shuffle(1000)
    data = data.batch(batch_size)
    data = data.prefetch(20)
    return data


dataset = load_tfrecord(datasets_to_use.pop(0))
# i = 0
# for y,x in dataset:
#     print(i)
#     i+=1
#     print(tf.keras.utils.to_categorical(x,num_classes=num_classes))


# concatenate the datasets to train the GAN on all the data
# while len(datasets_to_use)>0:
#     concatenate_dataset = load_tfrecord(datasets_to_use.pop(0))
#     dataset = dataset.concatenate(concatenate_dataset)


dataset = dataset.apply(prepare_dataset)
#dataset = dataset.take(10)

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

# Instantiate the customer `GANMonitor` Keras callback.
cbk = model.GANMonitor(image_dir=image_dir, num_classes=num_classes, num_img=3,
                       latent_dim=latent_dim, save_frequency=save_frequency)

# Get the wgan model
wgan = model.ConditionalWGAN(discriminator=discriminator, generator=generator,
                             latent_dim=latent_dim, image_size=image_size,
                             num_classes=num_classes,
                             discriminator_extra_steps=discriminator_extra_steps, )

# Compile the wgan model
wgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

checkpoint_filepath = os.path.join(checkpoint_dir,
                                   'checkpoint-{epoch:02d}.weights.h5')

# Regular expression to match the checkpoint pattern and extract the epoch number
pattern = re.compile(r"checkpoint-(\d+).weights.h5")

# Get the latest checkpoint file path
latest_checkpoint_path, s_epoch = get_latest_checkpoint(checkpoint_dir, pattern)

if latest_checkpoint_path:
    start_epoch = s_epoch
    # Load the model from the latest checkpoint
    wgan.load_weights(latest_checkpoint_path)
    print(f"Loaded model weights from {latest_checkpoint_path}")
else:
    print("No checkpoint found.")

print(wgan.metrics_names)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, monitor='generator_loss',
        save_weights_only=True, mode='min', verbose=1, save_best_only=False, save_freq=save_frequency)

print("Fitting the GAN")
wgan.fit(dataset, batch_size=batch_size, epochs=EPOCHS+start_epoch, initial_epoch=start_epoch,
         callbacks=[tensorboard_callback, model_checkpoint_callback, cbk])
