import os
import argparse
import ast

import tensorflow as tf
import tensorflow.keras as keras

import model


def generate_images(generator, to_generate, latent_dim, image_dir):
    for class_to_generate, number_to_generate in to_generate.items():
        one_hot_labels = tf.fill([number_to_generate],
                                 value=class_to_generate)  # generate class labels
        one_hot_labels = tf.one_hot(one_hot_labels, depth=num_classes)

        random_latent_vectors = keras.random.normal(
                shape=(number_to_generate, latent_dim),
                seed=keras.random.SeedGenerator(1337))
        print("one hot label shape", one_hot_labels.shape, "random shape",
              random_latent_vectors.shape)
        random_vector_labels = tf.keras.ops.concatenate(
                [random_latent_vectors, one_hot_labels], axis=1)

        # Decode the noise (guided by labels) to fake images.
        generated_images = generator(random_vector_labels, training=False)

        generated_images = (generated_images * 127.5) + 127.5
        generated_images = tf.cast(generated_images, tf.uint8)



        # make sure directories to save to exist
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(os.path.join(image_dir, str(class_to_generate).zfill(3)),
                    exist_ok=True)

        for i in range(number_to_generate):
            img = generated_images[i].numpy()
            print(img)
            img = keras.utils.array_to_img(img)
            print(img)
            padded_index = str(class_to_generate).zfill(3)  # Zero pad to 3 digits
            image_path = os.path.join(image_dir, padded_index,
                                      "{i}.png")  # directories for each class
            print("saving images to path")
            img.save(image_path.format(i=i))


# Define the default values
num_channels = 3
num_classes = 55
image_size = 224
latent_dim = 128
image_dir = "images_generated"
checkpoint_path = '../ckpt/checkpoint-10.weights.h5'

# define classes and numbers of samples to generate
# this is a dictionary with classes as keys and numbers of samples for the classes as values
to_generate = {1: 10, 3: 3, }

# Set up the argument parser
parser = argparse.ArgumentParser(description="Script for generating synthetic images")
parser.add_argument('--num_channels', type=int, default=num_channels,
                    help='Number of channels in the images (default: {})'.format(num_channels))
parser.add_argument('--num_classes', type=int, default=num_classes,
                    help='Number of classes in the dataset (default: {})'.format(num_classes))
parser.add_argument('--image_size', type=int, default=image_size,
                    help='Size of the images (default: {})'.format(image_size))
parser.add_argument('--latent_dim', type=int, default=latent_dim,
                    help='Latent dimension for the model (default: {})'.format(latent_dim))
parser.add_argument('--image_dir', type=str, default=image_dir,
                    help='Directory to store generated images (default: {})'.format(image_dir))
parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path,
                    help='Path to the model weights to load (default: {})'.format(checkpoint_path))
parser.add_argument('--to_generate', type=str, default=str(to_generate),
                    help='Dictionary indicating how much data to sample for each class (default: {})'.format(to_generate))


# Parse the command-line arguments
args = parser.parse_args()
# Convert the to_generate string back to a dictionary
to_generate = ast.literal_eval(args.to_generate)

print("Number of Channels:", args.num_channels)
print("Number of Classes:", args.num_classes)
print("Image Size:", args.image_size)
print("Latent Dimension:", args.latent_dim)
print("Image Directory:", args.image_dir)
print("Checkpoint Path:", args.checkpoint_path)
print("Samples to Generate:", to_generate)

# [TODO] for the first stage generate images for the minority classes so that the
#  whole dataset can be used for training in the next stage and is balanced with these samples

# [TODO] for the complete synthetic data sample from the second stage GAN
# sample as much data as we need for the highest percentage of synthetic data to train the classifier


# set up the model
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

# Get the wgan model
wgan = model.ConditionalWGAN(discriminator=discriminator, generator=generator,
                             latent_dim=latent_dim, image_size=image_size,
                             num_classes=num_classes,
                             discriminator_extra_steps=3, )

# Compile the wgan model
wgan.compile(d_optimizer=discriminator_optimizer,
             g_optimizer=generator_optimizer, g_loss_fn=generator_loss,
             d_loss_fn=discriminator_loss, )

if checkpoint_path:
    # Load the model from the latest checkpoint
    wgan.load_weights(checkpoint_path)
    print(f"Loaded model weights from {checkpoint_path}")
else:
    raise ValueError("No checkpoint found.")

generate_images(generator=wgan.generator, to_generate=to_generate,
                latent_dim=latent_dim, image_dir=image_dir)
