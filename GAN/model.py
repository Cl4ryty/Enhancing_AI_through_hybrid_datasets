import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import ops


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
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.image_size = image_size
        self.num_classes = num_classes

    def get_config(self):
        base_config = super().get_config()
        config = {"discriminator": tf.keras.utils.serialize_keras_object(
                self.discriminator),
                "generator": tf.keras.utils.serialize_keras_object(
                        self.generator),
                "latent_dim": tf.keras.utils.serialize_keras_object(
                        self.latent_dim),
                "discriminator_extra_steps": tf.keras.utils.serialize_keras_object(
                        self.d_steps), }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config["discriminator"] = tf.keras.utils.deserialize_keras_object(
                config["discriminator"])
        config["generator"] = tf.keras.utils.deserialize_keras_object(
                config["generator"])
        config["latent_dim"] = tf.keras.utils.deserialize_keras_object(
                config["latent_dim"])
        config[
            "discriminator_extra_steps"] = tf.keras.utils.deserialize_keras_object(
                config["discriminator_extra_steps"])
        return cls(**config)

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = ops.repeat(image_one_hot_labels,
                repeats=[self.image_size * self.image_size])
        image_one_hot_labels = ops.reshape(image_one_hot_labels,
                (-1, self.image_size, self.image_size, self.num_classes))

        for i in range(self.d_steps):
            # Sample random points in the latent space and concatenate the labels.
            # This is for the generator.
            batch_size = ops.shape(real_images)[0]
            random_latent_vectors = keras.random.normal(
                    shape=(batch_size, self.latent_dim),
                    seed=self.seed_generator)
            random_vector_labels = ops.concatenate(
                    [random_latent_vectors, one_hot_labels], axis=1)

            # Decode the noise (guided by labels) to fake images.
            generated_images = self.generator(random_vector_labels,
                                              training=False)

            # Combine them with real images. Note that we are concatenating the labels
            # with these images here.
            fake_image_and_labels = ops.concatenate(
                    [generated_images, image_one_hot_labels], -1)
            real_image_and_labels = ops.concatenate(
                    [real_images, image_one_hot_labels], -1)

            with tf.GradientTape() as tape:
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_image_and_labels,
                                                 training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_image_and_labels,
                                                 training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits,
                                        fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_image_and_labels,
                                           fake_image_and_labels)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss,
                                       self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                    zip(d_gradient, self.discriminator.trainable_variables))

        # Sample random points in the latent space.
        random_latent_vectors = keras.random.normal(
                shape=(batch_size, self.latent_dim), seed=self.seed_generator)
        random_vector_labels = ops.concatenate(
                [random_latent_vectors, one_hot_labels], axis=1)

        # Assemble labels that say "all real images".
        misleading_labels = ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels, training=True)
            fake_image_and_labels = ops.concatenate(
                    [fake_images, image_one_hot_labels], -1)
            predicted_labels = self.discriminator(fake_image_and_labels,
                                                  training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(misleading_labels, predicted_labels)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
                zip(gen_gradient, self.generator.trainable_variables))

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

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim))

        one_hot_labels = tf.fill([self.num_img],
                                 value=1)  # always generate images for class 1
        one_hot_labels = tf.one_hot(one_hot_labels, depth=self.num_classes)

        random_latent_vectors = keras.random.normal(
                shape=(self.num_img, self.latent_dim),
                seed=keras.random.SeedGenerator(1337))
        print("one hot label shape", one_hot_labels.shape, "random shape",
              random_latent_vectors.shape)
        random_vector_labels = ops.concatenate(
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
