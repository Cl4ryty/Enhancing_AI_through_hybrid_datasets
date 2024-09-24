import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt

import model
from utils import load_tfrecord


def generate_images(generator, class_to_generate, number_to_generate,
                    latent_dim, num_classes):
    one_hot_labels = tf.fill([number_to_generate],
                             value=class_to_generate)  # generate class labels
    one_hot_labels = tf.one_hot(one_hot_labels, depth=num_classes)

    random_latent_vectors = tf.random.normal(
            shape=(number_to_generate, latent_dim))
    print("one hot label shape", one_hot_labels.shape, "random shape",
          random_latent_vectors.shape)
    random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels],
                                     axis=1)

    # Decode the noise (guided by labels) to fake images.
    generated_images = generator(random_vector_labels, training=False)

    generated_images = (generated_images * 127.5) + 127.5
    generated_images = tf.cast(generated_images, tf.uint8)

    return generated_images


def get_unique_classes(dataset):
    class_labels = set()
    for _, label in dataset:
        class_labels.add(label.numpy())
    return sorted(
            list(class_labels))  # Return sorted list of unique class labels


def get_class_samples(dataset, class_label, num_samples):
    # Filter dataset for a specific class label
    filtered_dataset = dataset.filter(
            lambda image, label: tf.equal(label, class_label))

    # Take a certain number of samples
    samples = list(filtered_dataset.take(num_samples).as_numpy_iterator())
    return samples


def plot_images(real_images, generated_images, num_samples, class_label):
    plt.figure(figsize=(15, 10))

    # Display real images
    for i in range(num_samples):
        plt.subplot(5, num_samples, i + 1)  # First row for real images
        plt.imshow(real_images[i][
                       0])  # Accessing the first element as (image, label)
        plt.axis('off')
        plt.title('Real')

    # Display generated images
    for i in range(4 * num_samples):
        plt.subplot(5, num_samples,
                    i + 1 + num_samples)  # Second row for generated images
        plt.imshow(generated_images[i].numpy())
        plt.axis('off')
        plt.title('Generated')

    plt.suptitle(f'Class: {class_label}')
    plt.tight_layout()
    plt.show()


# Define the default values
num_channels = 3
num_classes = 55
image_size = 224
latent_dim = 128
image_dir = "images_generated"
checkpoint_path = '/media/hannah/DATA/A0_Uni/master/S2/Enhancing_AI/checkpoints/cgan/checkpoint-6274.weights.h5'
dataset_path = '../processed_datasets/train_balanced_undersampled.tfrecord'
num_samples = 5

dataset = load_tfrecord(dataset_path)

# Get unique class labels
unique_classes = get_unique_classes(dataset)

# set up the model
generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

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

print("unique classes", len(unique_classes), unique_classes)
for class_label in unique_classes:
    real_images = get_class_samples(dataset, class_label, num_samples)

    print("class_label", class_label)
    print(len(real_images))

    # Generate images
    generated_images = generate_images(generator, class_label, 4 * num_samples,
                                       latent_dim, num_classes)

    # Plot real and generated images
    plot_images(real_images, generated_images, num_samples, class_label)
