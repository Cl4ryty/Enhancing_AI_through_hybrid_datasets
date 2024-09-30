import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow.keras as keras

import model


def generate_images(generator, to_generate, latent_dim, image_dir):
    for class_to_generate, number_to_generate in to_generate.items():
        one_hot_labels = tf.fill([number_to_generate],
                                 value=1)  # always generate images for class 1
        one_hot_labels = tf.one_hot(one_hot_labels, depth=num_classes)

        random_latent_vectors = tf.random.normal(
                shape=(number_to_generate, latent_dim))
        print("one hot label shape", one_hot_labels.shape, "random shape",
              random_latent_vectors.shape)
        random_vector_labels = tf.concat(
                [random_latent_vectors, one_hot_labels], axis=1)

        # Decode the noise (guided by labels) to fake images.
        generated_images = generator(random_vector_labels, training=False)

        # make sure directories to save to exist
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(os.path.join(image_dir, str(class_to_generate).zfill(3)),
                    exist_ok=True)

        for i in range(number_to_generate):
            img = generated_images[i].numpy()
            img = keras.utils.array_to_img(img)
            padded_index = str(class_to_generate).zfill(
                3)  # Zero pad to 3 digits
            image_path = os.path.join(image_dir, padded_index,
                                      "{i}.png")  # directories for each class
            img.save(image_path.format(i=i))


# Define the default values
num_channels = 3
num_classes = 55
image_size = 224
latent_dim = 128
image_dir = "synthetic_100b"
checkpoint_path = '/media/hannah/DATA/A0_Uni/master/S2/Enhancing_AI/checkpoints/ncgan_s2/checkpoint-8384.weights.h5'

# define classes and numbers of samples to generate
# this is a dictionary with classes as keys and numbers of samples for the classes as values
to_generate = {1: 185, 2: 184, 3: 184, 4: 184, 5: 184, 8: 184, 9: 184, 11: 184, 14: 184, 15: 184, 18: 184, 19: 184, 20: 184, 21: 184, 23: 184, 24: 184, 26: 184, 27: 184, 28: 184, 29: 184, 30: 184, 31: 184, 32: 184, 33: 184, 34: 184, 35: 184, 36: 184, 37: 184, 38: 184, 39: 184, 40: 184, 41: 184, 42: 184, 43: 184, 44: 184, 45: 184, 46: 184, 47: 184, 48: 184, 49: 184, 50: 184, 51: 184, 52: 184, 54: 184}
# set up the model
generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes

# Create the discriminator.
discriminator = model.get_discriminator_model(image_size,
                                              discriminator_in_channels)
# Create the generator.
generator = model.get_generator_model(generator_in_channels)

# Get the wgan model
wgan = model.ConditionalWGAN(discriminator=discriminator, generator=generator,
                             latent_dim=latent_dim, image_size=image_size,
                             num_classes=num_classes,
                             discriminator_extra_steps=3, )

# Compile the wgan model
wgan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True), )

if checkpoint_path:
    # Load the model from the latest checkpoint
    wgan.load_weights(checkpoint_path)
    print(f"Loaded model weights from {checkpoint_path}")
else:
    raise ValueError("No checkpoint found.")

generate_images(generator=wgan.generator, to_generate=to_generate,
                latent_dim=latent_dim, image_dir=image_dir)
