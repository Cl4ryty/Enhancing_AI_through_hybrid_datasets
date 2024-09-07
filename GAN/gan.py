import os
import re
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras

import model


batch_size = 8
num_channels = 3
num_classes = 55
image_size = 224
latent_dim = 128
image_dir = "../images"
checkpoint_dir = '../ckpt'
EPOCHS = 10


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

def prepare_dataset(data):
    """
    Input Pipeline which prepares the dataset for further processing
    :param data: the dataset
    :return: preprocessed dataset
    """

    data = data.map(lambda image, target: (image, tf.keras.utils.to_categorical(target, num_classes=num_classes)))
    data = data.map(lambda image, target: (tf.cast(image, tf.float32)/255.0, target))
    # data = data.cache()
    data = data.shuffle(1000)
    data = data.batch(batch_size)
    data = data.prefetch(20)
    return data


def get_latest_checkpoint(checkpoint_dir):
    # List all files in the checkpoint directory
    files = os.listdir(checkpoint_dir)

    # Regular expression to match the checkpoint pattern and extract the epoch number
    pattern = re.compile(r"checkpoint-(\d+).weights.h5")

    # List to hold matching files with extracted epoch number
    checkpoint_files = []

    for file in files:
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            checkpoint_files.append((epoch, os.path.join(checkpoint_dir, file)))

    # Sort the checkpoint files by epoch number
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        # Get the latest checkpoint file path
        latest_checkpoint_path = checkpoint_files[0][1]
        return latest_checkpoint_path
    else:
        return None

train_dataset = load_tfrecord('../train1.tfrecord')
validation_dataset = load_tfrecord('../validation1.tfrecord')

# concatenate the datasets to train the GAN on all the data
dataset = train_dataset.concatenate(validation_dataset)
dataset = dataset.apply(prepare_dataset)
dataset = dataset.take(2)

# [TODO] for the first stage undersample all classes to the smallest number of samples of a class to train on a balanced dataset

# After loading the dataset and applying the above operations
for images, labels in dataset.take(1):  # Take just one batch to check the shape
    print("Batch Size (Images):", images.shape)  # Shape will be (batch_size, height, width, channels)
    print("Batch Size (Labels):", labels.shape)  # Shape will be (batch_size,)


generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)


# Create the discriminator.
discriminator =  model.get_discriminator_model(image_size, discriminator_in_channels)
# Create the generator.
generator = model.get_generator_model(generator_in_channels)

print("discriminator", discriminator.summary())
print("generator", generator.summary())


# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)


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
cbk = model.GANMonitor(image_dir=image_dir, num_classes=num_classes, num_img=3, latent_dim=latent_dim)

# Get the wgan model
wgan = model.ConditionalWGAN(
    discriminator=discriminator,
    generator=generator,
    latent_dim=latent_dim,
    image_size=image_size,
    num_classes=num_classes,
    discriminator_extra_steps=3,
)

# Compile the wgan model
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

checkpoint_filepath = os.path.join(checkpoint_dir, 'checkpoint-{epoch:02d}.weights.h5')

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
        filepath=checkpoint_filepath,
        monitor='generator_loss',
        save_weights_only=True,
        mode='min',
        verbose=1,
        save_best_only=False)

print("Fitting the GAN")
wgan.fit(dataset, batch_size=batch_size, epochs=EPOCHS, callbacks=[tensorboard_callback, model_checkpoint_callback, cbk])