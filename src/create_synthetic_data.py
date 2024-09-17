import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
image_dir = "minority_synthetic_data2"
checkpoint_path = '/media/hannah/DATA/A0_Uni/master/S2/Enhancing_AI/checkpoints/cgan/checkpoint-6375.weights.h5'

# define classes and numbers of samples to generate
# this is a dictionary with classes as keys and numbers of samples for the classes as values
to_generate = {1: 24, 2: 0, 3: 0, 4: 127, 5: 0, 8: 151, 9: 60, 11: 160, 14: 0,
               15: 55, 18: 0, 19: 0, 20: 153, 21: 0, 23: 0, 24: 0, 26: 90,
               27: 116, 28: 153, 29: 150, 30: 152, 31: 131, 32: 127, 33: 44,
               34: 28, 35: 70, 36: 139, 37: 114, 38: 157, 39: 140, 40: 108,
               41: 155, 42: 120, 43: 99, 44: 70, 45: 20, 46: 103, 47: 148,
               48: 0, 49: 71, 50: 138, 51: 96, 52: 109, 54: 80}

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
