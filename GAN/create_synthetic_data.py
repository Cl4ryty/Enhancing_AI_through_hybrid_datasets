import os
import re
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras

import model



def generate_images(generator, to_generate, latent_dim, image_dir):
    for class_to_generate, number_to_generate in to_generate.items():
        one_hot_labels = tf.fill([number_to_generate],
                                 value=class_to_generate)  # generate class labels
        one_hot_labels = tf.one_hot(one_hot_labels, depth=num_classes)

        random_latent_vectors = keras.random.normal(
                shape=(number_to_generate,latent_dim),
                seed=keras.random.SeedGenerator(1337))
        print("one hot label shape", one_hot_labels.shape, "random shape",
              random_latent_vectors.shape)
        random_vector_labels = tf.keras.ops.concatenate(
                [random_latent_vectors, one_hot_labels], axis=1)

        # Decode the noise (guided by labels) to fake images.
        generated_images = generator(random_vector_labels,
                                                training=False)

        generated_images = (generated_images * 127.5) + 127.5

        # make sure directories to save to exist
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(os.path.join(image_dir, str(class_to_generate)), exist_ok=True)

        for i in range(number_to_generate):
            img = generated_images[i].numpy()
            img = keras.utils.array_to_img(img)
            image_path = os.path.join(image_dir, str(class_to_generate) , "{i}.png") # directories for each class
            print("saving images to path")
            img.save(image_path.format(i=i))


num_channels = 3
num_classes = 55
image_size = 224
latent_dim = 128
# define the path to store the generated data
image_dir = "images_generated"
# define the path of the model weights to load
checkpoint_path = '../ckpt/checkpoint-10.weights.h5'

# define classes and numbers of samples to generate
# this is a dictionary with classes as keys and numbers of samples for the classes as values
# [TODO] for the first stage generate images for the minority classes so that the
#  whole dataset can be used for training in the next stage and is balanced with these samples

# [TODO] for the complete synthetic data sample from the second stage GAN
# sample as much data as we need for the highest percentage of synthetic data to train the classifier
to_generate = {
        1:10,
        3:3,
}


# set up the model
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

if checkpoint_path:
    # Load the model from the latest checkpoint
    wgan.load_weights(checkpoint_path)
    print(f"Loaded model weights from {checkpoint_path}")
else:
    raise ValueError("No checkpoint found.")


generate_images(generator=wgan.generator,
                to_generate=to_generate,
                latent_dim=latent_dim,
                image_dir=image_dir)