import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def conv_block(x, filters, activation, kernel_size=(3, 3), strides=(1, 1),
        padding="same", use_bias=True, use_bn=False, use_dropout=False,
        drop_value=0.5, ):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
            use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def upsample_block(x, filters, activation, kernel_size=(3, 3), strides=(1, 1),
        up_size=(2, 2), padding="same", use_bn=False, use_bias=True,
        use_dropout=False, drop_value=0.3, ):
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
            use_bias=use_bias)(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_discriminator_model(image_size, discriminator_in_channels):
    img_input = layers.Input(
        shape=(image_size, image_size, discriminator_in_channels))
    # Zero pad the input to make the input images size to (32, 32, 1).
    x = conv_block(img_input, 64, kernel_size=(5, 5), strides=(2, 2),
            use_bn=False, use_bias=True, activation=layers.LeakyReLU(0.2),
            use_dropout=False, drop_value=0.3, )
    x = conv_block(x, 128, kernel_size=(5, 5), strides=(2, 2), use_bn=False,
            activation=layers.LeakyReLU(0.2), use_bias=True, use_dropout=True,
            drop_value=0.3, )
    x = conv_block(x, 256, kernel_size=(5, 5), strides=(2, 2), use_bn=False,
            activation=layers.LeakyReLU(0.2), use_bias=True, use_dropout=True,
            drop_value=0.3, )
    x = conv_block(x, 512, kernel_size=(5, 5), strides=(2, 2), use_bn=False,
            activation=layers.LeakyReLU(0.2), use_bias=True, use_dropout=False,
            drop_value=0.3, )

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)

    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


def get_generator_model(generator_in_channels):
    noise = layers.Input(shape=(generator_in_channels,))
    x = layers.Dense(14 * 14 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((14, 14, 256))(x)
    x = upsample_block(x, 128, layers.LeakyReLU(0.2), strides=(1, 1),
            use_bias=False, use_bn=True, padding="same", use_dropout=False, )
    x = upsample_block(x, 64, layers.LeakyReLU(0.2), strides=(1, 1),
            use_bias=False, use_bn=True, padding="same", use_dropout=False, )
    x = upsample_block(x, 32, layers.LeakyReLU(0.2), strides=(1, 1),
                       use_bias=False, use_bn=True, padding="same",
                       use_dropout=False, )

    x = upsample_block(x, 3, layers.Activation("tanh"), strides=(1, 1),
            use_bias=False, use_bn=True)

    g_model = keras.models.Model(noise, x, name="generator")
    return g_model


@tf.keras.utils.register_keras_serializable(package="WGAN")
class ConditionalWGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, image_size,
                 num_classes, discriminator_extra_steps=3, gp_weight=10.0):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.image_size = image_size
        self.num_classes = num_classes

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(image_one_hot_labels,
                repeats=[self.image_size * self.image_size])
        image_one_hot_labels = tf.reshape(image_one_hot_labels,
                (-1, self.image_size, self.image_size, self.num_classes))

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
                [random_latent_vectors, one_hot_labels], axis=1)

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat(
                [generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat(
                [real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
                [fake_image_and_labels, real_image_and_labels], axis=0)

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights))

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
                [random_latent_vectors, one_hot_labels], axis=1)

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat(
                    [fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {"g_loss": self.gen_loss_tracker.result(),
                "d_loss": self.disc_loss_tracker.result(), }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, image_dir, num_classes, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim))

        one_hot_labels = tf.fill([self.num_img],
                                 value=1)  # always generate images for class 1
        one_hot_labels = tf.one_hot(one_hot_labels, depth=self.num_classes)

        random_latent_vectors = tf.random.normal(
                shape=(self.num_img, self.latent_dim))
        print("one hot label shape", one_hot_labels.shape, "random shape",
              random_latent_vectors.shape)
        random_vector_labels = tf.concat(
                [random_latent_vectors, one_hot_labels], axis=1)

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.model.generator(random_vector_labels,
                                                training=False)

        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.utils.array_to_img(img)
            image_path = os.path.join(self.image_dir,
                                      "generated_img_{i}_{epoch}.png")
            print("saving images to path")
            img.save(image_path.format(i=i, epoch=epoch))
